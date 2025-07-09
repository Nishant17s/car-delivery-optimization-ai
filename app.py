import matplotlib
matplotlib.use('Agg')  # Fix RuntimeError issue

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import joblib
import os
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def generate_bar_chart(predictions):
    df = pd.DataFrame(predictions, columns=['Company', 'Manufacturer', 'Model', 'Year', 'Sales'])
    
    plt.figure(figsize=(10, 6))
    plt.style.use("dark_background")
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')  # Light black background
    
    neon_palette = ["#ff00ff", "#00ffff", "#ff9900", "#33ff00", "#ff0000", "#6600cc"]  # Neon colors
    bars = sns.barplot(x='Year', y='Sales', hue='Company', data=df, dodge=True, palette=neon_palette)
    
    plt.xlabel('Year', fontsize=12, fontweight='bold', color='white')
    plt.ylabel('Predicted Sales', fontsize=12, fontweight='bold', color='white')
    plt.title('Predicted Car Sales for Next 5 Years', fontsize=14, fontweight='bold', color='white')
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    plt.legend(title='Company', loc='upper left', facecolor='#2a2a2a', edgecolor='white')
    
    # Adding interactive tooltips
    for bar, (company, sales) in zip(bars.patches, zip(df['Company'], df['Sales'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{company}\n{sales}',
                ha='center', va='bottom', fontsize=10, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor='#1e1e1e')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f'data:image/png;base64,{graph_url}'

@app.route('/', methods=['GET', 'POST'])
def home():
    predictions, csv_preview, csv_headers, insights, graph_url = [], [], [], [], None
    error = None
    
    if request.method == 'POST':
        file = request.files['file']
        
        if file:
            try:
                df = pd.read_csv(file)
                df.columns = [col.strip().replace(" ", "_") for col in df.columns]
                
                required_columns = {"Company", "Manufacturer", "Model", "Year", "Sales"}
                if not required_columns.issubset(df.columns):
                    error = "❌ Error: CSV must contain Company, Manufacturer, Model, Year, and Sales columns."
                    return render_template_string(template, error=error)
                
                csv_headers = df.columns.tolist()
                csv_preview = df.head(5).values.tolist()
                
                X = df[["Year"]]
                y = df["Sales"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                future_years = np.array([[year] for year in range(df["Year"].max() + 1, df["Year"].max() + 6)])
                predicted_sales = model.predict(future_years)
                
                company_info = df[['Company', 'Manufacturer', 'Model']].drop_duplicates().values.tolist()
                predictions = [(company, manufacturer, model, year, int(sales)) 
                               for (company, manufacturer, model), (year, sales) in zip(company_info, zip(future_years.flatten(), predicted_sales))]
                
                avg_growth = df["Sales"].pct_change().mean() * 100
                insights.append(f"📈 Avg. Sales Growth: {avg_growth:.2f}% per year.")
                insights.append(f"🚗 Best-selling Model: {df.groupby('Model')['Sales'].sum().idxmax()}.")
                insights.append(f"🏆 Top Company: {df.groupby('Company')['Sales'].sum().idxmax()}.")
                insights.append(f"🏭 Production Recommendation: Increase production of top-selling models to meet demand.")
                insights.append(f"💡 Strategy: Invest in marketing for low-selling models or consider discontinuation.")
                
                graph_url = generate_bar_chart(predictions)
                
            except Exception as e:
                error = f"❌ Error processing file: {str(e)}"
    
    return render_template_string(template, csv_headers=csv_headers, csv_preview=csv_preview, predictions=predictions, insights=insights, graph_url=graph_url, error=error)
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Innovators - AI Car Sales Predictions</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #121212; font-family: 'Century Gothic', sans-serif; color: white; text-align: center; }
        .container { margin-top: 50px; background: #1e1e1e; padding: 30px; border-radius: 12px; box-shadow: 3px 3px 15px rgba(0,0,0,0.3); }
        .brand { font-size: 28px; font-weight: bold; color: cyan; text-shadow: 0px 0px 10px cyan; }
        .btn-container { display: flex; justify-content: center; gap: 15px; margin: 20px 0; }
        .btn { transition: all 0.3s ease-in-out; }
        .btn:hover { transform: scale(1.05); box-shadow: 0px 0px 12px rgba(255, 255, 255, 0.5); }
        .content-table { display: flex; justify-content: space-between; gap: 20px; flex-wrap: wrap; }
        .table-container { flex: 1; background: #2a2a2a; padding: 20px; border-radius: 8px; box-shadow: 3px 3px 10px rgba(0,0,0,0.3); }
        table { width: 100%; margin-top: 10px; border-collapse: collapse; }
        th, td { padding: 10px; border: 1px solid white; text-align: left; }
        th { background: cyan; color: black; font-weight: bold; }
        tr:nth-child(even) { background: #3a3a3a; }
        .hidden { display: none; }
        img { max-width: 100%; height: auto; border-radius: 10px; margin-top: 15px; }
        .graph-container { margin-top: 20px; background: #2a2a2a; padding: 20px; border-radius: 10px; }
    </style>
    <script>
        function toggleSection(id) {
            var section = document.getElementById(id);
            section.style.display = (section.style.display === "none" || section.style.display === "") ? "block" : "none";
        }
    </script>
</head>
<body>
<div class="container">
    <h1 class="brand">AutoML Innovators</h1>
    
    <form method="post" enctype="multipart/form-data">
        <label for="file" class="form-label">📂 Upload CSV File:</label>
        <input type="file" class="form-control mb-3" name="file" accept=".csv" required>
        <button type="submit" class="btn btn-primary">Upload & Analyze</button>
    </form>

    <div class="btn-container">
        <button class="btn btn-warning" onclick="toggleSection('dataSection')">📊 Show Data & Insights</button>
        <button class="btn btn-info" onclick="toggleSection('graphSection')">📈 Show Graph</button>
    </div>

    <!-- CSV Preview & Business Insights in Side-by-Side Table Layout -->
    <div id="dataSection" class="hidden">
        <div class="content-table">
            <div class="table-container">
                <h3>📃 CSV Preview</h3>
                <table>
                    <tr>
                        {% for header in csv_headers %}
                            <th>{{ header }}</th>
                        {% endfor %}
                    </tr>
                    {% for row in csv_preview %}
                        <tr>
                            {% for cell in row %}
                                <td>{{ cell }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </table>
            </div>

            <div class="table-container">
                <h3>📊 Business Insights</h3>
                <table>
                    {% for insight in insights %}
                        <tr><td>{{ insight }}</td></tr>
                    {% endfor %}
                </table>
            </div>
        </div>
    </div>

    <!-- Graph Section -->
    <div id="graphSection" class="hidden graph-container">
        <h3>Predicted Sales Graph</h3>
        <img src="{{ graph_url }}" alt="Predicted Sales Graph">
    </div>
</div>
</body>
</html>

"""


if __name__ == "__main__":
    app.run(debug=True)
