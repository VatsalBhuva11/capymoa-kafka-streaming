"""
Live Visualization Dashboard for Streaming ML Pipeline
Monitors CSV logs from Kafka consumers and displays real-time metrics.
Supports multi-model comparison and automated updates.
"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import os
import glob
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np


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
        """Setup dashboard layout with dark mode."""
        # Dark mode color scheme
        bg_dark = '#0f1419'
        card_dark = '#1a1f2e'
        text_primary = '#e4e6eb'
        text_secondary = '#b0b3b8'
        accent_blue = '#3b82f6'
        accent_green = '#10b981'
        accent_red = '#ef4444'
        accent_purple = '#8b5cf6'
        accent_orange = '#f59e0b'
        border_color = '#2d3748'
        
        # Add CSS styles using app.index_string
        self.app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
                /* Override default browser styles for inputs and dropdowns */
                input[type="number"], input[type="text"] {
                    background-color: #1a1f2e !important;
                    color: #e4e6eb !important;
                    border: 1px solid #2d3748 !important;
                    outline: none !important;
                }
                
                input[type="number"]:focus, input[type="text"]:focus {
                    border: 1px solid #3b82f6 !important;
                    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1) !important;
                }
                
                /* Dropdown styling - Dash uses react-select */
                .Select-control, .Select-menu-outer, 
                .css-1s2u09g-control, .css-1pahdxg-control,
                .css-26l3qy-menu {
                    background-color: #1a1f2e !important;
                    border: 1px solid #2d3748 !important;
                    color: #e4e6eb !important;
                    box-shadow: none !important;
                }
                
                .Select-control:hover, 
                .css-1s2u09g-control:hover,
                .css-1pahdxg-control:hover {
                    border: 1px solid #3b82f6 !important;
                }
                
                .Select-value-label, .Select-input input,
                .css-1uccc91-singleValue, .css-1hwfws3 {
                    color: #e4e6eb !important;
                }
                
                .Select-placeholder, .css-1wa3eu0-placeholder {
                    color: #b0b3b8 !important;
                }
                
                .Select-option, .css-1n7v3ny-option {
                    background-color: #1a1f2e !important;
                    color: #e4e6eb !important;
                }
                
                .Select-option:hover, .Select-option.is-focused,
                .css-1n7v3ny-option:hover, .css-1n7v3ny-option--is-focused {
                    background-color: #2d3748 !important;
                }
                
                .Select-option.is-selected, .css-1n7v3ny-option--is-selected {
                    background-color: #3b82f6 !important;
                    color: #ffffff !important;
                }
                
                /* Remove white outline/border from focused elements */
                .Select-control:focus, .Select-control:focus-within,
                .css-1pahdxg-control:focus, .css-1pahdxg-control:focus-within {
                    border: 1px solid #3b82f6 !important;
                    box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.1) !important;
                    outline: none !important;
                }
                
                /* Remove white borders from all elements */
                * {
                    box-sizing: border-box;
                }
                
                /* Remove any default white borders/outlines - but keep focus indicators */
                input:focus, select:focus, button:focus {
                    outline: none !important;
                }
                
                /* Ensure no white backgrounds leak through */
                body, html {
                    background-color: #0f1419 !important;
                }
                
                /* Checklist styling */
                .form-check-input {
                    background-color: #1a1f2e !important;
                    border: 1px solid #2d3748 !important;
                }
                
                .form-check-input:checked {
                    background-color: #3b82f6 !important;
                    border-color: #3b82f6 !important;
                }
                
                /* Additional react-select overrides for all possible class names */
                [class*="control"], [class*="menu"], [class*="option"] {
                    background-color: #1a1f2e !important;
                    border-color: #2d3748 !important;
                    color: #e4e6eb !important;
                }
                
                /* Ensure input text is visible */
                input::placeholder {
                    color: #b0b3b8 !important;
                    opacity: 0.7 !important;
                }
                
                /* Make sure number input arrows are visible */
                input[type="number"]::-webkit-inner-spin-button,
                input[type="number"]::-webkit-outer-spin-button {
                    opacity: 0.7;
                    filter: invert(0.5);
                }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
        
        self.app.layout = html.Div([
            # Header section
            html.Div([
                html.Div([
                    html.H1("Streaming ML Pipeline Dashboard", 
                            style={
                                'textAlign': 'left', 
                                'color': text_primary, 
                                'marginBottom': '8px',
                                'fontSize': '2.2em',
                                'fontWeight': '600',
                                'letterSpacing': '-0.5px'
                            }),
                    html.P("Real-time Machine Learning Performance Monitoring", 
                           style={
                               'textAlign': 'left', 
                               'color': text_secondary, 
                               'fontSize': '0.95em', 
                               'marginBottom': '0',
                               'fontWeight': '400'
                           }),
                ], style={'flex': '1'}),
                html.Div([
                    html.Div([
                        html.Label("Auto-refresh (s):", 
                                  style={
                                      'fontWeight': '500', 
                                      'color': text_secondary, 
                                      'marginRight': '10px',
                                      'fontSize': '0.9em'
                                  }),
                        dcc.Input(
                            id='refresh-interval',
                            type='number',
                            value=self.update_interval,
                            min=1,
                            max=60,
                            style={
                                'width': '70px', 
                                'padding': '8px 12px',
                                'borderRadius': '6px',
                                'border': f'1px solid {border_color}',
                                'backgroundColor': card_dark,
                                'color': text_primary,
                                'fontSize': '0.95em',
                                'fontWeight': '600',
                                'outline': 'none',
                                'boxShadow': 'none',
                                'caretColor': accent_blue
                            }
                        ),
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Button("Refresh Now", id='refresh-button', n_clicks=0,
                                   style={
                                       'padding': '8px 20px', 
                                       'backgroundColor': accent_blue, 
                                       'color': '#ffffff',
                                       'border': 'none', 
                                       'borderRadius': '6px',
                                       'cursor': 'pointer',
                                       'fontWeight': '500',
                                       'fontSize': '0.9em',
                                       'transition': 'all 0.2s'
                                   }),
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Span("Last updated: ", 
                                 style={
                                     'fontWeight': '500', 
                                     'color': text_secondary, 
                                     'marginRight': '5px',
                                     'fontSize': '0.9em'
                                 }),
                        html.Span(id='last-update-time', children='Never',
                                 style={
                                     'color': accent_green, 
                                     'fontFamily': 'monospace',
                                     'fontSize': '0.9em'
                                 })
                    ], style={'display': 'inline-block'}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'padding': '24px 32px',
                'backgroundColor': card_dark,
                'borderRadius': '12px',
                'marginBottom': '24px',
                'border': f'1px solid {border_color}'
            }),
            
            # Filters section
            html.Div([
                html.Div([
                    html.Label("Task Type", 
                              style={
                                  'fontWeight': '500', 
                                  'marginBottom': '8px', 
                                  'color': text_secondary,
                                  'fontSize': '0.85em',
                                  'textTransform': 'uppercase',
                                  'letterSpacing': '0.5px'
                              }),
                    dcc.Dropdown(
                        id='task-filter',
                        options=[
                            {'label': 'All Tasks', 'value': 'all'},
                            {'label': 'Classification', 'value': 'classification'},
                            {'label': 'Regression', 'value': 'regression'}
                        ],
                        value='all',
                        style={
                            'width': '100%'
                        },
                        className='dark-dropdown'
                    ),
                ], style={'flex': '1', 'marginRight': '16px'}),
                html.Div([
                    html.Label("Dataset", 
                              style={
                                  'fontWeight': '500', 
                                  'marginBottom': '8px', 
                                  'color': text_secondary,
                                  'fontSize': '0.85em',
                                  'textTransform': 'uppercase',
                                  'letterSpacing': '0.5px'
                              }),
                    dcc.Dropdown(
                        id='dataset-filter',
                        options=[],
                        value='all',
                        style={
                            'width': '100%'
                        },
                        className='dark-dropdown'
                    ),
                ], style={'flex': '1', 'marginRight': '16px'}),
                html.Div([
                    html.Label("Models", 
                              style={
                                  'fontWeight': '500', 
                                  'marginBottom': '8px', 
                                  'color': text_secondary,
                                  'fontSize': '0.85em',
                                  'textTransform': 'uppercase',
                                  'letterSpacing': '0.5px'
                              }),
                    dcc.Checklist(
                        id='model-checklist',
                        options=[],
                        value=[],
                        inline=True,
                        style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px'}
                    ),
                ], style={'flex': '2'}),
            ], style={
                'display': 'flex',
                'marginBottom': '24px', 
                'padding': '20px', 
                'backgroundColor': card_dark,
                'borderRadius': '12px', 
                'border': f'1px solid {border_color}'
            }),
            
            # Real-time statistics cards
            html.Div(id='realtime-stats', style={'marginBottom': '24px'}),
            
            # Metrics summary cards
            html.Div(id='summary-cards', style={'marginBottom': '24px'}),
            
            # Main plots
            html.Div([
                html.Div([
                    dcc.Graph(id='cumulative-metrics-plot', style={'height': '450px'}),
                ], style={
                    'marginBottom': '24px', 
                    'padding': '20px', 
                    'backgroundColor': card_dark,
                    'borderRadius': '12px', 
                    'border': f'1px solid {border_color}'
                }),
                
                html.Div([
                    dcc.Graph(id='window-metrics-plot', style={'height': '450px'}),
                ], style={
                    'marginBottom': '24px', 
                    'padding': '20px', 
                    'backgroundColor': card_dark,
                    'borderRadius': '12px', 
                    'border': f'1px solid {border_color}'
                }),
                
                # Drift events plot
                html.Div([
                    dcc.Graph(id='drift-events-plot', style={'height': '350px'}),
                ], style={
                    'marginBottom': '24px', 
                    'padding': '20px', 
                    'backgroundColor': card_dark,
                    'borderRadius': '12px', 
                    'border': f'1px solid {border_color}'
                }),
            ]),
            
            # Data table
            html.Div([
                html.H3("Latest Metrics", 
                       style={
                           'marginBottom': '16px', 
                           'color': text_primary,
                           'fontSize': '1.3em',
                           'fontWeight': '600',
                           'borderBottom': f'2px solid {border_color}',
                           'paddingBottom': '12px'
                       }),
                html.Div(id='metrics-table'),
            ], style={
                'padding': '24px', 
                'backgroundColor': card_dark,
                'borderRadius': '12px', 
                'border': f'1px solid {border_color}'
            }),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,
                n_intervals=0
            ),
            
            # Store for data and previous update time
            dcc.Store(id='metrics-store'),
            dcc.Store(id='previous-update-time'),
            dcc.Store(id='previous-instance-count'),
        ], style={
            'padding': '24px', 
            'backgroundColor': bg_dark, 
            'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
            'minHeight': '100vh',
            'color': text_primary
        })
    
    def create_realtime_stats(self, df, previous_time=None, previous_instance_count=None):
        """Create real-time statistics cards with dark mode."""
        if df.empty:
            return html.Div(
                "No data available. Start running consumers with --log-file option.",
                style={'color': '#b0b3b8', 'padding': '20px', 'textAlign': 'center'}
            )
        
        # Dark mode colors
        card_dark = '#1a1f2e'
        text_primary = '#e4e6eb'
        text_secondary = '#b0b3b8'
        accent_blue = '#3b82f6'
        accent_green = '#10b981'
        accent_red = '#ef4444'
        accent_purple = '#8b5cf6'
        accent_orange = '#f59e0b'
        border_color = '#2d3748'
        
        cards = []
        
        # Calculate total instances processed
        total_instances = df['instance'].max() if 'instance' in df.columns else 0
        
        # Calculate processing rate (instances per second)
        processing_rate = 0
        if previous_time is not None and previous_instance_count is not None:
            try:
                time_diff = (datetime.now() - datetime.fromisoformat(previous_time)).total_seconds()
                if time_diff > 0 and total_instances > previous_instance_count:
                    instances_diff = total_instances - previous_instance_count
                    processing_rate = instances_diff / time_diff
            except:
                pass
        
        # Count drift events
        drift_count = 0
        files = glob.glob(os.path.join(self.results_dir, '*_results.json'))
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    results = json.load(f)
                    if 'drift_details' in results:
                        drift_count += len(results['drift_details'])
            except:
                pass
        
        # Calculate average accuracy/MAE across all models
        latest_data = df.groupby(['dataset', 'task', 'model']).last().reset_index()
        avg_accuracy = None
        if not latest_data.empty:
            class_data = latest_data[latest_data['task'] == 'classification']
            if not class_data.empty and 'cumulative_accuracy' in class_data.columns:
                acc_values = pd.to_numeric(class_data['cumulative_accuracy'].replace('N/A', None), errors='coerce')
                if acc_values.notna().any():
                    avg_accuracy = acc_values.mean()
        
        # Count active models
        active_models = len(latest_data['model'].unique()) if not latest_data.empty else 0
        
        # Create stat cards
        stat_cards = [
            {
                'title': 'Total Instances',
                'value': f"{int(total_instances):,}",
                'subtitle': 'Processed',
                'color': accent_blue
            },
            {
                'title': 'Processing Rate',
                'value': f"{processing_rate:.1f}",
                'subtitle': 'instances/sec',
                'color': accent_green
            },
            {
                'title': 'Drift Events',
                'value': f"{drift_count}",
                'subtitle': 'Detected',
                'color': accent_red
            },
            {
                'title': 'Avg Accuracy',
                'value': f"{avg_accuracy:.3f}" if avg_accuracy is not None else "N/A",
                'subtitle': 'Across Models',
                'color': accent_purple
            },
            {
                'title': 'Active Models',
                'value': f"{active_models}",
                'subtitle': 'Running',
                'color': accent_orange
            }
        ]
        
        for stat in stat_cards:
            cards.append(
                html.Div([
                    html.Div([
                        html.H4(stat['title'], 
                               style={
                                   'margin': '0 0 12px 0', 
                                   'color': text_secondary, 
                                   'fontSize': '0.75em', 
                                   'fontWeight': '600',
                                   'textTransform': 'uppercase',
                                   'letterSpacing': '0.5px'
                               }),
                        html.H2(stat['value'], 
                               style={
                                   'margin': '0 0 8px 0', 
                                   'color': stat['color'],
                                   'fontSize': '2.2em',
                                   'fontWeight': '700',
                                   'letterSpacing': '-0.5px'
                               }),
                        html.P(stat['subtitle'], 
                              style={
                                  'margin': '0', 
                                  'color': text_secondary, 
                                  'fontSize': '0.85em',
                                  'fontWeight': '400'
                              }),
                    ], style={'textAlign': 'left'})
                ], style={
                    'backgroundColor': card_dark,
                    'padding': '24px',
                    'borderRadius': '12px',
                    'textAlign': 'left',
                    'flex': '1',
                    'margin': '0 8px',
                    'borderLeft': f'4px solid {stat["color"]}',
                    'border': f'1px solid {border_color}',
                    'minWidth': '180px'
                })
            )
        
        return html.Div(cards, style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap', 'gap': '16px'})
    
    def create_summary_cards(self, df):
        """Create summary metric cards with dark mode."""
        if df.empty:
            return html.Div(
                "No data available. Start running consumers with --log-file option.",
                style={'color': '#b0b3b8', 'padding': '20px', 'textAlign': 'center'}
            )
        
        # Dark mode colors
        card_dark = '#1a1f2e'
        text_primary = '#e4e6eb'
        text_secondary = '#b0b3b8'
        accent_green = '#10b981'
        accent_red = '#ef4444'
        border_color = '#2d3748'
        
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
            
            color = accent_green if task_type == 'classification' else accent_red
            
            cards.append(
                html.Div([
                    html.H4(f"{task_type.capitalize()} - Best {metric_name}", 
                           style={
                               'margin': '0 0 12px 0', 
                               'color': text_secondary, 
                               'fontSize': '0.75em', 
                               'fontWeight': '600',
                               'textTransform': 'uppercase',
                               'letterSpacing': '0.5px'
                           }),
                    html.P(f"{best_row['model']} on {best_row['dataset']}", 
                          style={
                              'margin': '0 0 16px 0', 
                              'color': text_secondary, 
                              'fontSize': '0.9em',
                              'fontWeight': '400'
                          }),
                    html.H2(f"{best_value:.4f}" if best_value != 'N/A' else 'N/A', 
                           style={
                               'margin': '0', 
                               'color': color,
                               'fontSize': '2.4em',
                               'fontWeight': '700',
                               'letterSpacing': '-0.5px'
                           }),
                ], style={
                    'backgroundColor': card_dark,
                    'padding': '24px',
                    'borderRadius': '12px',
                    'textAlign': 'left',
                    'flex': '1',
                    'margin': '0 8px',
                    'borderLeft': f'4px solid {color}',
                    'border': f'1px solid {border_color}',
                    'minWidth': '280px'
                })
            )
        
        if not cards:
            return html.Div(
                "No metrics available yet.",
                style={'color': '#b0b3b8', 'padding': '20px', 'textAlign': 'center'}
            )
        
        return html.Div(cards, style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap', 'gap': '16px'})
    
    def create_cumulative_plot(self, df, selected_models):
        """Create cumulative metrics plot with improved styling."""
        if df.empty:
            return {
                'data': [], 
                'layout': {
                    'title': {'text': 'No data available', 'font': {'size': 18}},
                    'template': 'plotly_white'
                }
            }
        
        filtered_df = df[df['model'].isin(selected_models)] if selected_models else df
        
        traces = []
        colors = px.colors.qualitative.Set3
        
        # Classification plots
        class_df = filtered_df[filtered_df['task'] == 'classification']
        if not class_df.empty and 'cumulative_accuracy' in class_df.columns:
            for idx, ((dataset, model), group) in enumerate(class_df.groupby(['dataset', 'model'])):
                group = group.copy()
                group['cumulative_accuracy'] = pd.to_numeric(
                    group['cumulative_accuracy'].replace('N/A', None), errors='coerce'
                )
                group = group.dropna(subset=['cumulative_accuracy'])
                group = group.sort_values('instance')
                
                if not group.empty:
                    color = colors[idx % len(colors)]
                    # Convert hex to rgba for fill
                    def hex_to_rgba(hex_color, alpha):
                        hex_color = hex_color.lstrip('#')
                        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        return f'rgba({r}, {g}, {b}, {alpha})'
                    traces.append(go.Scatter(
                        x=group['instance'],
                        y=group['cumulative_accuracy'],
                        mode='lines+markers',
                        name=f"{model} ({dataset})",
                        line=dict(width=3, color=color),
                        marker=dict(size=5, color=color, line=dict(width=1, color='white')),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                     'Instance: %{x:,}<br>' +
                                     'Accuracy: %{y:.4f}<extra></extra>',
                        fill='tonexty' if idx > 0 else None,
                        fillcolor=hex_to_rgba(color, 0.1) if idx > 0 else None
                    ))
        
        # Regression plots
        reg_df = filtered_df[filtered_df['task'] == 'regression']
        if not reg_df.empty and 'cumulative_mae' in reg_df.columns:
            for idx, ((dataset, model), group) in enumerate(reg_df.groupby(['dataset', 'model'])):
                group = group.copy()
                group['cumulative_mae'] = pd.to_numeric(
                    group['cumulative_mae'].replace('N/A', None), errors='coerce'
                )
                group = group.dropna(subset=['cumulative_mae'])
                group = group.sort_values('instance')
                
                if not group.empty:
                    color = colors[(idx + len(class_df.groupby(['dataset', 'model']))) % len(colors)]
                    traces.append(go.Scatter(
                        x=group['instance'],
                        y=group['cumulative_mae'],
                        mode='lines+markers',
                        name=f"{model} ({dataset}) - MAE",
                        line=dict(width=3, color=color, dash='dash'),
                        marker=dict(size=5, color=color, line=dict(width=1, color='white')),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                     'Instance: %{x:,}<br>' +
                                     'MAE: %{y:.4f}<extra></extra>',
                        yaxis='y2'
                    ))
        
        layout = go.Layout(
            title={
                'text': 'Cumulative Metrics Over Time',
                'font': {'size': 18, 'color': '#e4e6eb'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title=dict(text='Instance Number', font=dict(size=13, color='#b0b3b8')),
                gridcolor='rgba(255,255,255,0.05)',
                showgrid=True,
                zeroline=False,
                color='#b0b3b8'
            ),
            yaxis=dict(
                title=dict(text='Cumulative Accuracy', font=dict(size=13, color='#b0b3b8')),
                side='left',
                range=[0, 1] if not class_df.empty else None,
                gridcolor='rgba(255,255,255,0.05)',
                showgrid=True,
                zeroline=False,
                color='#b0b3b8'
            ),
            yaxis2=dict(
                title=dict(text='Cumulative MAE', font=dict(size=13, color='#b0b3b8')),
                side='right',
                overlaying='y',
                range=[0, None] if not reg_df.empty else None,
                gridcolor='rgba(255,255,255,0.02)',
                showgrid=False,
                color='#b0b3b8'
            ),
            hovermode='x unified',
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(26, 31, 46, 0.95)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1,
                font=dict(color='#e4e6eb', size=11)
            ),
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=60, r=100, t=50, b=60),
            font=dict(color='#e4e6eb')
        )
        
        return {'data': traces, 'layout': layout}
    
    def create_window_plot(self, df, selected_models):
        """Create window/rolling metrics plot with improved styling."""
        if df.empty:
            return {
                'data': [], 
                'layout': {
                    'title': {'text': 'No data available', 'font': {'size': 18}},
                    'template': 'plotly_white'
                }
            }
        
        filtered_df = df[df['model'].isin(selected_models)] if selected_models else df
        
        traces = []
        colors = px.colors.qualitative.Set3
        
        # Classification plots
        class_df = filtered_df[filtered_df['task'] == 'classification']
        if not class_df.empty and 'window_accuracy' in class_df.columns:
            for idx, ((dataset, model), group) in enumerate(class_df.groupby(['dataset', 'model'])):
                group = group.copy()
                group['window_accuracy'] = pd.to_numeric(
                    group['window_accuracy'].replace('N/A', None), errors='coerce'
                )
                group = group.dropna(subset=['window_accuracy'])
                group = group.sort_values('instance')
                
                if not group.empty:
                    color = colors[idx % len(colors)]
                    # Convert hex to rgba for fill
                    def hex_to_rgba(hex_color, alpha):
                        hex_color = hex_color.lstrip('#')
                        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        return f'rgba({r}, {g}, {b}, {alpha})'
                    traces.append(go.Scatter(
                        x=group['instance'],
                        y=group['window_accuracy'],
                        mode='lines+markers',
                        name=f"{model} ({dataset})",
                        line=dict(width=2.5, color=color),
                        marker=dict(size=4, color=color, line=dict(width=0.5, color='white')),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                     'Instance: %{x:,}<br>' +
                                     'Window Accuracy: %{y:.4f}<extra></extra>',
                        fill='tonexty' if idx > 0 else None,
                        fillcolor=hex_to_rgba(color, 0.15) if idx > 0 else None
                    ))
        
        # Regression plots
        reg_df = filtered_df[filtered_df['task'] == 'regression']
        if not reg_df.empty and 'window_mae' in reg_df.columns:
            for idx, ((dataset, model), group) in enumerate(reg_df.groupby(['dataset', 'model'])):
                group = group.copy()
                group['window_mae'] = pd.to_numeric(
                    group['window_mae'].replace('N/A', None), errors='coerce'
                )
                group = group.dropna(subset=['window_mae'])
                group = group.sort_values('instance')
                
                if not group.empty:
                    color = colors[(idx + len(class_df.groupby(['dataset', 'model']))) % len(colors)]
                    traces.append(go.Scatter(
                        x=group['instance'],
                        y=group['window_mae'],
                        mode='lines+markers',
                        name=f"{model} ({dataset}) - Window MAE",
                        line=dict(width=2.5, color=color, dash='dash'),
                        marker=dict(size=4, color=color, line=dict(width=0.5, color='white')),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                     'Instance: %{x:,}<br>' +
                                     'Window MAE: %{y:.4f}<extra></extra>',
                        yaxis='y2'
                    ))
        
        layout = go.Layout(
            title={
                'text': 'Rolling Window Metrics (Last 1000 instances)',
                'font': {'size': 18, 'color': '#e4e6eb'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title=dict(text='Instance Number', font=dict(size=13, color='#b0b3b8')),
                gridcolor='rgba(255,255,255,0.05)',
                showgrid=True,
                zeroline=False,
                color='#b0b3b8'
            ),
            yaxis=dict(
                title=dict(text='Window Accuracy', font=dict(size=13, color='#b0b3b8')),
                side='left',
                range=[0, 1] if not class_df.empty else None,
                gridcolor='rgba(255,255,255,0.05)',
                showgrid=True,
                zeroline=False,
                color='#b0b3b8'
            ),
            yaxis2=dict(
                title=dict(text='Window MAE', font=dict(size=13, color='#b0b3b8')),
                side='right',
                overlaying='y',
                range=[0, None] if not reg_df.empty else None,
                gridcolor='rgba(255,255,255,0.02)',
                showgrid=False,
                color='#b0b3b8'
            ),
            hovermode='x unified',
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(26, 31, 46, 0.95)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1,
                font=dict(color='#e4e6eb', size=11)
            ),
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=60, r=100, t=50, b=60),
            font=dict(color='#e4e6eb')
        )
        
        return {'data': traces, 'layout': layout}
    
    def create_drift_plot(self, df, selected_models):
        """Create drift events visualization with improved styling."""
        if df.empty:
            return {
                'data': [], 
                'layout': {
                    'title': {'text': 'No data available', 'font': {'size': 18}},
                    'template': 'plotly_white'
                }
            }
        
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
            return {
                'data': [], 
                'layout': {
                    'title': {
                        'text': 'No drift events detected yet',
                        'font': {'size': 18, 'color': '#e4e6eb'}
                    },
                    'template': 'plotly_dark',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'font': dict(color='#e4e6eb')
                }
            }
        
        drift_df = pd.DataFrame(drift_data)
        colors = px.colors.qualitative.Set3
        
        for idx, model in enumerate(drift_df['model'].unique()):
            model_drifts = drift_df[drift_df['model'] == model]
            color = colors[idx % len(colors)]
            traces.append(go.Scatter(
                x=model_drifts['instance'],
                y=[model] * len(model_drifts),
                mode='markers',
                name=model,
                marker=dict(
                    size=15, 
                    symbol='triangle-down', 
                    color=color,
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Instance: %{x:,}<br>' +
                             'Drift Detected<extra></extra>'
            ))
        
        layout = go.Layout(
            title={
                'text': 'Concept Drift Events',
                'font': {'size': 18, 'color': '#e4e6eb'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title=dict(text='Instance Number', font=dict(size=13, color='#b0b3b8')),
                gridcolor='rgba(255,255,255,0.05)',
                showgrid=True,
                zeroline=False,
                color='#b0b3b8'
            ),
            yaxis=dict(
                title=dict(text='Model', font=dict(size=13, color='#b0b3b8')),
                gridcolor='rgba(255,255,255,0.05)',
                showgrid=True,
                zeroline=False,
                color='#b0b3b8'
            ),
            hovermode='closest',
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(26, 31, 46, 0.95)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1,
                font=dict(color='#e4e6eb', size=11)
            ),
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=60, r=100, t=50, b=60),
            font=dict(color='#e4e6eb')
        )
        
        return {'data': traces, 'layout': layout}
    
    def create_metrics_table(self, df):
        """Create metrics data table with dark mode."""
        if df.empty:
            return html.Div(
                "No data available.",
                style={'color': '#b0b3b8', 'padding': '20px', 'textAlign': 'center'}
            )
        
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
            return html.Div(
                "No metrics available.",
                style={'color': '#b0b3b8', 'padding': '20px', 'textAlign': 'center'}
            )
        
        table_df = pd.DataFrame(table_data)
        
        # Dark mode colors
        card_dark = '#1a1f2e'
        text_primary = '#e4e6eb'
        text_secondary = '#b0b3b8'
        border_color = '#2d3748'
        header_bg = '#2d3748'
        
        return html.Div([
            html.Table([
                html.Thead([
                    html.Tr([html.Th(col, style={
                        'padding': '12px 16px', 
                        'border': f'1px solid {border_color}', 
                        'backgroundColor': header_bg, 
                        'color': text_primary,
                        'fontWeight': '600',
                        'fontSize': '0.85em',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px',
                        'textAlign': 'left'
                    }) 
                    for col in table_df.columns])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(table_df.iloc[i][col], style={
                            'padding': '12px 16px', 
                            'border': f'1px solid {border_color}',
                            'color': text_primary,
                            'fontSize': '0.9em'
                        }) 
                        for col in table_df.columns
                    ], style={
                        'backgroundColor': card_dark if i % 2 == 0 else '#1f2533'
                    }) 
                    for i in range(len(table_df))
                ])
            ], style={
                'width': '100%', 
                'borderCollapse': 'collapse', 
                'fontSize': '14px',
                'borderRadius': '8px',
                'overflow': 'hidden'
            })
        ], style={'overflowX': 'auto'})
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('metrics-store', 'data'),
             Output('last-update-time', 'children'),
             Output('previous-update-time', 'data'),
             Output('previous-instance-count', 'data'),
             Output('dataset-filter', 'options'),
             Output('model-checklist', 'options')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks'),
             Input('refresh-interval', 'value')],
            [State('task-filter', 'value'),
             State('previous-update-time', 'data'),
             State('previous-instance-count', 'data')]
        )
        def update_data(n_intervals, n_clicks, refresh_interval, task_filter, previous_time, previous_instance_count):
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
            previous_timestamp = previous_time if previous_time else timestamp
            
            # Get current max instance count - this will be the "previous" for next update
            current_instance_count = df['instance'].max() if not df.empty and 'instance' in df.columns else 0
            # Store current as previous for next callback
            new_previous_count = current_instance_count
            
            return data_json, timestamp, previous_timestamp, new_previous_count, dataset_options, model_options
        
        @self.app.callback(
            Output('interval-component', 'interval'),
            [Input('refresh-interval', 'value')]
        )
        def update_interval(interval):
            """Update refresh interval."""
            return (interval or self.update_interval) * 1000
        
        @self.app.callback(
            Output('realtime-stats', 'children'),
            [Input('metrics-store', 'data')],
            [State('previous-update-time', 'data'),
             State('previous-instance-count', 'data')]
        )
        def update_realtime_stats(data, previous_time, previous_instance_count):
            """Update real-time statistics."""
            if not data:
                return html.Div(
                    "No data available.",
                    style={'color': '#b0b3b8', 'padding': '20px', 'textAlign': 'center'}
                )
            df = pd.DataFrame(data)
            return self.create_realtime_stats(df, previous_time, previous_instance_count)
        
        @self.app.callback(
            Output('summary-cards', 'children'),
            [Input('metrics-store', 'data')]
        )
        def update_summary_cards(data):
            """Update summary cards."""
            if not data:
                return html.Div(
                    "No data available.",
                    style={'color': '#b0b3b8', 'padding': '20px', 'textAlign': 'center'}
                )
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
                return {
                    'data': [], 
                    'layout': {
                        'title': {'text': 'No data available', 'font': {'size': 18, 'color': '#e4e6eb'}},
                        'template': 'plotly_dark',
                        'plot_bgcolor': 'rgba(0,0,0,0)',
                        'paper_bgcolor': 'rgba(0,0,0,0)',
                        'font': dict(color='#e4e6eb')
                    }
                }
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
                return {
                    'data': [], 
                    'layout': {
                        'title': {'text': 'No data available', 'font': {'size': 18, 'color': '#e4e6eb'}},
                        'template': 'plotly_dark',
                        'plot_bgcolor': 'rgba(0,0,0,0)',
                        'paper_bgcolor': 'rgba(0,0,0,0)',
                        'font': dict(color='#e4e6eb')
                    }
                }
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
                return {
                    'data': [], 
                    'layout': {
                        'title': {'text': 'No data available', 'font': {'size': 18, 'color': '#e4e6eb'}},
                        'template': 'plotly_dark',
                        'plot_bgcolor': 'rgba(0,0,0,0)',
                        'paper_bgcolor': 'rgba(0,0,0,0)',
                        'font': dict(color='#e4e6eb')
                    }
                }
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
                return html.Div(
                    "No data available.",
                    style={'color': '#b0b3b8', 'padding': '20px', 'textAlign': 'center'}
                )
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

