# styles.py

def get_css():
    return """
    <style>
        /* Main container */
        .reportview-container .main .block-container {
            max-width: 1200px;
            padding: 2rem 1rem;
            background-color: #f0f2f5;
        }

        /* Dashboard grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        /* Metric cards */
        .metric-card {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }

        .metric-card h3 {
            color: #666;
            font-size: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .metric-card p {
            color: #333;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .metric-card .trend {
            font-size: 0.85rem;
            font-weight: 600;
        }

        .metric-card .icon {
            float: right;
            font-size: 2.5rem;
            color: #e0e0e0;
        }

        /* Trend colors */
        .trend-up { color: #28a745; }
        .trend-down { color: #dc3545; }

        /* Charts */
        .chart-container {
            background-color: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }

        .chart-container h3 {
            color: #444;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        /* Tables */
        .dataframe {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }

        .dataframe th, .dataframe td {
            border: none;
            padding: 0.75rem;
            text-align: left;
        }

        .dataframe thead th {
            background-color: #f8f9fa;
            color: #495057;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
        }

        .dataframe tbody tr:nth-of-type(even) {
            background-color: #f8f9fa;
        }

        .dataframe tbody tr:hover {
            background-color: #e9ecef;
        }

        /* Custom scrollbar for webkit browsers */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
    """
def metric_card_html(title, value, trend=None, trend_suffix="from last month", icon="📊", chart_html=""):
    trend_html = ""
    if trend is not None:
        trend_class = "trend-up" if trend >= 0 else "trend-down"
        trend_symbol = "↑" if trend >= 0 else "↓"
        trend_html = f'<div class="metric-card-trend {trend_class}">{trend_symbol} {abs(trend)}% {trend_suffix}</div>'

    return f"""
    <div class="metric-card">
        <div class="metric-card-header">
            <div class="metric-card-title">{title}</div>
            <div class="metric-card-icon">{icon}</div>
        </div>
        <div class="metric-card-value">{value}</div>
        {trend_html}
        <div class="metric-card-chart">{chart_html}</div>
    </div>
    """

def create_dashboard_grid(metrics):
    grid_html = '<div class="dashboard-grid">'
    for metric in metrics:
        grid_html += metric_card_html(**metric)
    grid_html += '</div>'
    return grid_html

def chart_container(title, chart_html):
    return f"""
    <div class="chart-container">
        <h3>{title}</h3>
        {chart_html}
    </div>
    """