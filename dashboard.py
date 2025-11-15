"""
Live Visualization Dashboard for Streaming ML Pipeline
Monitors CSV logs from Kafka consumers and displays real-time metrics.
Supports multi-model comparison and automated updates.
"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import os
import glob
import time
from datetime import datetime
from pathlib import Path
import json


class MetricsDashboard:
    """Dashboard for visualizing streaming ML metrics."""
    
    def __init__(self, results_dir='results', update_interval=2):
        """
        Initialize dashboard.
        
        Args:
            results_dir: Directory containing CSV log files
            update_interval: Seconds between auto-refreshes
        """
        self.results_dir = results_dir
        self.update_interval = update_interval
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def find_csv_files(self):
        """Find all CSV metric files in results directory."""
        if not os.path.exists(self.results_dir):
            return []
        
        pattern = os.path.join(self.results_dir, '*_metrics.csv')
        files = glob.glob(pattern)
        return sorted(files)
    
    def parse_filename(self, filepath):
        """Extract dataset, task, and model from filename."""
        filename = os.path.basename(filepath)
        # Format: {dataset}_{task}_{model}_metrics.csv
        parts = filename.replace('_metrics.csv', '').split('_')
        if len(parts) >= 3:
            dataset = parts[0]
            task = parts[1]
            model = '_'.join(parts[2:])  # Handle models with underscores
            return dataset, task, model
        return None, None, None
    
    def load_metrics(self, filepath):
        """Load metrics from CSV file."""
        try:
            df = pd.read_csv(filepath)
            dataset, task, model = self.parse_filename(filepath)
            df['dataset'] = dataset
            df['task'] = task
            df['model'] = model
            df['file'] = os.path.basename(filepath)
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_all_metrics(self):
        """Load all available metric files."""
        files = self.find_csv_files()
        all_data = []
        
        for filepath in files:
            df = self.load_metrics(filepath)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.Div([
                html.H1("Streaming ML Pipeline - Live Dashboard", 
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Div([
                    html.Div([
                        html.Label("Auto-refresh (seconds):", style={'fontWeight': 'bold'}),
                        dcc.Input(
                            id='refresh-interval',
                            type='number',
                            value=self.update_interval,
                            min=1,
                            max=60,
                            style={'width': '100px', 'marginLeft': '10px'}
                        ),
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Button("Refresh Now", id='refresh-button', n_clicks=0,
                                   style={'padding': '10px 20px', 'backgroundColor': '#3498db', 
                                         'color': 'white', 'border': 'none', 'borderRadius': '5px',
                                         'cursor': 'pointer'}),
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Span("Last updated: ", style={'fontWeight': 'bold'}),
                        html.Span(id='last-update-time', children='Never')
                    ], style={'display': 'inline-block'}),
                ], style={'textAlign': 'center', 'marginBottom': '20px', 'padding': '10px',
                         'backgroundColor': '#ecf0f1', 'borderRadius': '5px'}),
            ]),
            
            html.Div([
                html.Div([
                    html.Label("Task Type:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='task-filter',
                        options=[
                            {'label': 'All Tasks', 'value': 'all'},
                            {'label': 'Classification', 'value': 'classification'},
                            {'label': 'Regression', 'value': 'regression'}
                        ],
                        value='all',
                        style={'width': '200px', 'display': 'inline-block'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                html.Div([
                    html.Label("Dataset:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='dataset-filter',
                        options=[],
                        value='all',
                        style={'width': '200px', 'display': 'inline-block'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                html.Div([
                    html.Label("Models:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Checklist(
                        id='model-checklist',
                        options=[],
                        value=[],
                        inline=True,
                        style={'display': 'inline-block'}
                    ),
                ], style={'display': 'inline-block'}),
            ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#ffffff',
                     'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Metrics summary cards
            html.Div(id='summary-cards', style={'marginBottom': '20px'}),
            
            # Main plots
            html.Div([
                dcc.Graph(id='cumulative-metrics-plot', style={'height': '400px'}),
            ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#ffffff',
                     'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                dcc.Graph(id='window-metrics-plot', style={'height': '400px'}),
            ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#ffffff',
                     'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Drift events plot
            html.Div([
                dcc.Graph(id='drift-events-plot', style={'height': '300px'}),
            ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#ffffff',
                     'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Data table
            html.Div([
                html.H3("Latest Metrics", style={'marginBottom': '10px'}),
                html.Div(id='metrics-table'),
            ], style={'padding': '15px', 'backgroundColor': '#ffffff',
                     'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,  # Convert to milliseconds
                n_intervals=0
            ),
            
            # Store for data
            dcc.Store(id='metrics-store'),
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'fontFamily': 'Arial, sans-serif'})
    
    def create_summary_cards(self, df):
        """Create summary metric cards."""
        if df.empty:
            return html.Div("No data available. Start running consumers with --log-file option.")
        
        cards = []
        
        # Filter for latest instance of each model
        latest_data = df.groupby(['dataset', 'task', 'model']).last().reset_index()
        
        for task_type in ['classification', 'regression']:
            task_data = latest_data[latest_data['task'] == task_type]
            if task_data.empty:
                continue
            
            if task_type == 'classification':
                metric_col = 'cumulative_accuracy'
                metric_name = 'Accuracy'
                # Convert to numeric, handling 'N/A' values
                numeric_values = pd.to_numeric(task_data[metric_col].replace('N/A', None), errors='coerce')
                if numeric_values.notna().any():
                    best_idx = numeric_values.idxmax()
                else:
                    continue
            else:
                metric_col = 'cumulative_mae'
                metric_name = 'MAE'
                # Convert to numeric, handling 'N/A' values
                numeric_values = pd.to_numeric(task_data[metric_col].replace('N/A', None), errors='coerce')
                if numeric_values.notna().any():
                    best_idx = numeric_values.idxmin()
                else:
                    continue
            
            best_row = task_data.loc[best_idx]
            best_value = best_row[metric_col]
            
            cards.append(
                html.Div([
                    html.H4(f"{task_type.capitalize()} - Best {metric_name}", 
                           style={'margin': '0', 'color': '#2c3e50'}),
                    html.P(f"{best_row['model']} on {best_row['dataset']}", 
                          style={'margin': '5px 0', 'color': '#7f8c8d', 'fontSize': '14px'}),
                    html.H2(f"{best_value:.4f}" if best_value != 'N/A' else 'N/A', 
                           style={'margin': '10px 0', 'color': '#27ae60' if task_type == 'classification' else '#e74c3c'}),
                ], style={
                    'backgroundColor': '#ffffff',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'textAlign': 'center',
                    'flex': '1',
                    'margin': '0 10px'
                })
            )
        
        if not cards:
            return html.Div("No metrics available yet.")
        
        return html.Div(cards, style={'display': 'flex', 'justifyContent': 'space-around'})
    
    def create_cumulative_plot(self, df, selected_models):
        """Create cumulative metrics plot."""
        if df.empty:
            return {'data': [], 'layout': {'title': 'No data available'}}
        
        filtered_df = df[df['model'].isin(selected_models)] if selected_models else df
        
        traces = []
        
        # Classification plots
        class_df = filtered_df[filtered_df['task'] == 'classification']
        if not class_df.empty and 'cumulative_accuracy' in class_df.columns:
            for (dataset, model), group in class_df.groupby(['dataset', 'model']):
                group = group.copy()
                group['cumulative_accuracy'] = pd.to_numeric(
                    group['cumulative_accuracy'].replace('N/A', None), errors='coerce'
                )
                group = group.dropna(subset=['cumulative_accuracy'])
                
                if not group.empty:
                    traces.append(go.Scatter(
                        x=group['instance'],
                        y=group['cumulative_accuracy'],
                        mode='lines+markers',
                        name=f"{model} ({dataset}) - Accuracy",
                        line=dict(width=2),
                        marker=dict(size=4)
                    ))
        
        # Regression plots
        reg_df = filtered_df[filtered_df['task'] == 'regression']
        if not reg_df.empty and 'cumulative_mae' in reg_df.columns:
            for (dataset, model), group in reg_df.groupby(['dataset', 'model']):
                group = group.copy()
                group['cumulative_mae'] = pd.to_numeric(
                    group['cumulative_mae'].replace('N/A', None), errors='coerce'
                )
                group = group.dropna(subset=['cumulative_mae'])
                
                if not group.empty:
                    traces.append(go.Scatter(
                        x=group['instance'],
                        y=group['cumulative_mae'],
                        mode='lines+markers',
                        name=f"{model} ({dataset}) - MAE",
                        line=dict(width=2),
                        marker=dict(size=4),
                        yaxis='y2'
                    ))
        
        layout = go.Layout(
            title='Cumulative Metrics Over Time',
            xaxis=dict(title='Instance Number'),
            yaxis=dict(title='Cumulative Accuracy', side='left', 
                      range=[0, 1] if class_df.empty else None),
            yaxis2=dict(title='Cumulative MAE', side='right', overlaying='y',
                       range=[0, None] if reg_df.empty else None),
            hovermode='closest',
            legend=dict(x=1.05, y=1),
            template='plotly_white'
        )
        
        return {'data': traces, 'layout': layout}
    
    def create_window_plot(self, df, selected_models):
        """Create window/rolling metrics plot."""
        if df.empty:
            return {'data': [], 'layout': {'title': 'No data available'}}
        
        filtered_df = df[df['model'].isin(selected_models)] if selected_models else df
        
        traces = []
        
        # Classification plots
        class_df = filtered_df[filtered_df['task'] == 'classification']
        if not class_df.empty and 'window_accuracy' in class_df.columns:
            for (dataset, model), group in class_df.groupby(['dataset', 'model']):
                group = group.copy()
                group['window_accuracy'] = pd.to_numeric(
                    group['window_accuracy'].replace('N/A', None), errors='coerce'
                )
                group = group.dropna(subset=['window_accuracy'])
                
                if not group.empty:
                    traces.append(go.Scatter(
                        x=group['instance'],
                        y=group['window_accuracy'],
                        mode='lines+markers',
                        name=f"{model} ({dataset}) - Window Accuracy",
                        line=dict(width=2),
                        marker=dict(size=4)
                    ))
        
        # Regression plots
        reg_df = filtered_df[filtered_df['task'] == 'regression']
        if not reg_df.empty and 'window_mae' in reg_df.columns:
            for (dataset, model), group in reg_df.groupby(['dataset', 'model']):
                group = group.copy()
                group['window_mae'] = pd.to_numeric(
                    group['window_mae'].replace('N/A', None), errors='coerce'
                )
                group = group.dropna(subset=['window_mae'])
                
                if not group.empty:
                    traces.append(go.Scatter(
                        x=group['instance'],
                        y=group['window_mae'],
                        mode='lines+markers',
                        name=f"{model} ({dataset}) - Window MAE",
                        line=dict(width=2),
                        marker=dict(size=4),
                        yaxis='y2'
                    ))
        
        layout = go.Layout(
            title='Rolling Window Metrics (Last 1000 instances)',
            xaxis=dict(title='Instance Number'),
            yaxis=dict(title='Window Accuracy', side='left',
                      range=[0, 1] if class_df.empty else None),
            yaxis2=dict(title='Window MAE', side='right', overlaying='y',
                       range=[0, None] if reg_df.empty else None),
            hovermode='closest',
            legend=dict(x=1.05, y=1),
            template='plotly_white'
        )
        
        return {'data': traces, 'layout': layout}
    
    def create_drift_plot(self, df, selected_models):
        """Create drift events visualization."""
        if df.empty:
            return {'data': [], 'layout': {'title': 'No data available'}}
        
        # Try to load drift events from JSON results files
        traces = []
        drift_data = []
        
        files = glob.glob(os.path.join(self.results_dir, '*_results.json'))
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    results = json.load(f)
                    if 'drift_details' in results and results['drift_details']:
                        filename = os.path.basename(filepath)
                        parts = filename.replace('_results.json', '').split('_')
                        if len(parts) >= 3:
                            model = '_'.join(parts[2:])
                            if not selected_models or model in selected_models:
                                for drift in results['drift_details']:
                                    drift_data.append({
                                        'instance': drift.get('instance_id', 0),
                                        'model': model,
                                        'timestamp': drift.get('timestamp', '')
                                    })
            except Exception as e:
                pass
        
        if not drift_data:
            return {'data': [], 'layout': {'title': 'No drift events detected yet'}}
        
        drift_df = pd.DataFrame(drift_data)
        
        for model in drift_df['model'].unique():
            model_drifts = drift_df[drift_df['model'] == model]
            traces.append(go.Scatter(
                x=model_drifts['instance'],
                y=[model] * len(model_drifts),
                mode='markers',
                name=model,
                marker=dict(size=10, symbol='triangle-down', color='red')
            ))
        
        layout = go.Layout(
            title='Concept Drift Events',
            xaxis=dict(title='Instance Number'),
            yaxis=dict(title='Model'),
            hovermode='closest',
            template='plotly_white'
        )
        
        return {'data': traces, 'layout': layout}
    
    def create_metrics_table(self, df):
        """Create metrics data table."""
        if df.empty:
            return html.Div("No data available.")
        
        # Get latest metrics for each model
        latest = df.groupby(['dataset', 'task', 'model']).last().reset_index()
        
        # Format table
        table_data = []
        for _, row in latest.iterrows():
            if row['task'] == 'classification':
                cum_acc = row.get('cumulative_accuracy', 'N/A')
                win_acc = row.get('window_accuracy', 'N/A')
                table_data.append({
                    'Dataset': row['dataset'],
                    'Task': row['task'],
                    'Model': row['model'],
                    'Instance': int(row['instance']) if pd.notna(row['instance']) else 0,
                    'Cumulative Accuracy': f"{cum_acc:.4f}" if cum_acc != 'N/A' and pd.notna(cum_acc) else 'N/A',
                    'Window Accuracy': f"{win_acc:.4f}" if win_acc != 'N/A' and pd.notna(win_acc) else 'N/A',
                })
            else:
                cum_mae = row.get('cumulative_mae', 'N/A')
                cum_mse = row.get('cumulative_mse', 'N/A')
                win_mae = row.get('window_mae', 'N/A')
                table_data.append({
                    'Dataset': row['dataset'],
                    'Task': row['task'],
                    'Model': row['model'],
                    'Instance': int(row['instance']) if pd.notna(row['instance']) else 0,
                    'Cumulative MAE': f"{cum_mae:.4f}" if cum_mae != 'N/A' and pd.notna(cum_mae) else 'N/A',
                    'Cumulative MSE': f"{cum_mse:.4f}" if cum_mse != 'N/A' and pd.notna(cum_mse) else 'N/A',
                    'Window MAE': f"{win_mae:.4f}" if win_mae != 'N/A' and pd.notna(win_mae) else 'N/A',
                })
        
        if not table_data:
            return html.Div("No metrics available.")
        
        table_df = pd.DataFrame(table_data)
        
        return html.Div([
            html.Table([
                html.Thead([
                    html.Tr([html.Th(col, style={'padding': '10px', 'border': '1px solid #ddd', 
                                                'backgroundColor': '#3498db', 'color': 'white'}) 
                            for col in table_df.columns])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(table_df.iloc[i][col], style={'padding': '10px', 'border': '1px solid #ddd'}) 
                        for col in table_df.columns
                    ], style={'backgroundColor': '#f2f2f2' if i % 2 == 0 else 'white'}) 
                    for i in range(len(table_df))
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '14px'})
        ], style={'overflowX': 'auto'})
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('metrics-store', 'data'),
             Output('last-update-time', 'children'),
             Output('dataset-filter', 'options'),
             Output('model-checklist', 'options')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks'),
             Input('refresh-interval', 'value')],
            [State('task-filter', 'value')]
        )
        def update_data(n_intervals, n_clicks, refresh_interval, task_filter):
            """Update metrics data."""
            df = self.load_all_metrics()
            
            # Update interval based on user input
            if refresh_interval and refresh_interval != self.update_interval:
                self.update_interval = refresh_interval
            
            # Filter by task
            if task_filter and task_filter != 'all':
                df = df[df['task'] == task_filter]
            
            # Get unique datasets and models
            datasets = ['all'] + sorted(df['dataset'].unique().tolist()) if not df.empty else ['all']
            models = sorted(df['model'].unique().tolist()) if not df.empty else []
            
            dataset_options = [{'label': d, 'value': d} for d in datasets]
            model_options = [{'label': m, 'value': m} for m in models]
            
            # Store data as JSON
            data_json = df.to_dict('records') if not df.empty else []
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return data_json, timestamp, dataset_options, model_options
        
        @self.app.callback(
            Output('interval-component', 'interval'),
            [Input('refresh-interval', 'value')]
        )
        def update_interval(interval):
            """Update refresh interval."""
            return (interval or self.update_interval) * 1000
        
        @self.app.callback(
            Output('summary-cards', 'children'),
            [Input('metrics-store', 'data')]
        )
        def update_summary_cards(data):
            """Update summary cards."""
            if not data:
                return html.Div("No data available.")
            df = pd.DataFrame(data)
            return self.create_summary_cards(df)
        
        @self.app.callback(
            Output('cumulative-metrics-plot', 'figure'),
            [Input('metrics-store', 'data'),
             Input('model-checklist', 'value'),
             Input('dataset-filter', 'value')]
        )
        def update_cumulative_plot(data, selected_models, selected_dataset):
            """Update cumulative metrics plot."""
            if not data:
                return {'data': [], 'layout': {'title': 'No data available'}}
            df = pd.DataFrame(data)
            if selected_dataset and selected_dataset != 'all':
                df = df[df['dataset'] == selected_dataset]
            return self.create_cumulative_plot(df, selected_models)
        
        @self.app.callback(
            Output('window-metrics-plot', 'figure'),
            [Input('metrics-store', 'data'),
             Input('model-checklist', 'value'),
             Input('dataset-filter', 'value')]
        )
        def update_window_plot(data, selected_models, selected_dataset):
            """Update window metrics plot."""
            if not data:
                return {'data': [], 'layout': {'title': 'No data available'}}
            df = pd.DataFrame(data)
            if selected_dataset and selected_dataset != 'all':
                df = df[df['dataset'] == selected_dataset]
            return self.create_window_plot(df, selected_models)
        
        @self.app.callback(
            Output('drift-events-plot', 'figure'),
            [Input('metrics-store', 'data'),
             Input('model-checklist', 'value')]
        )
        def update_drift_plot(data, selected_models):
            """Update drift events plot."""
            if not data:
                return {'data': [], 'layout': {'title': 'No data available'}}
            df = pd.DataFrame(data)
            return self.create_drift_plot(df, selected_models)
        
        @self.app.callback(
            Output('metrics-table', 'children'),
            [Input('metrics-store', 'data'),
             Input('dataset-filter', 'value')]
        )
        def update_metrics_table(data, selected_dataset):
            """Update metrics table."""
            if not data:
                return html.Div("No data available.")
            df = pd.DataFrame(data)
            if selected_dataset and selected_dataset != 'all':
                df = df[df['dataset'] == selected_dataset]
            return self.create_metrics_table(df)
    
    def run(self, host='127.0.0.1', port=8050, debug=False):
        """Run the dashboard server."""
        print(f"\n{'='*60}")
        print("Starting Streaming ML Dashboard")
        print(f"{'='*60}")
        print(f"Dashboard URL: http://{host}:{port}")
        print(f"Results directory: {self.results_dir}")
        print(f"Auto-refresh interval: {self.update_interval} seconds")
        print(f"{'='*60}\n")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live visualization dashboard for streaming ML pipeline')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing CSV log files (default: results)')
    parser.add_argument('--port', type=int, default=8050,
                       help='Port to run dashboard on (default: 8050)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--update-interval', type=int, default=2,
                       help='Seconds between auto-refreshes (default: 2)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    dashboard = MetricsDashboard(
        results_dir=args.results_dir,
        update_interval=args.update_interval
    )
    dashboard.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

