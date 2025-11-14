"""
Dashboard Module
Live visualization of metrics, drift alerts, and performance trends.
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import json
import time
from threading import Thread
from typing import Dict, List
import numpy as np


class StreamingDashboard:
    """Interactive dashboard for streaming ML pipeline."""
    
    def __init__(self, port: int = 8050):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.port = port
        self.metrics_data = {
            'classification': {'history': [], 'current': {}},
            'regression': {'history': [], 'current': {}}
        }
        self.drift_events = []
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Streaming ML Pipeline Dashboard", className="text-center mb-4"),
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Current Metrics"),
                        dbc.CardBody(id="current-metrics")
                    ], className="mb-4")
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Trends"),
                        dbc.CardBody([
                            dcc.Graph(id="metrics-plot")
                        ])
                    ])
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Drift Events"),
                        dbc.CardBody(id="drift-alerts")
                    ])
                ], width=12)
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=2*1000,  # Update every 2 seconds
                n_intervals=0
            )
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('current-metrics', 'children'),
             Output('metrics-plot', 'figure'),
             Output('drift-alerts', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Update current metrics display
            metrics_html = self._create_metrics_display()
            
            # Update metrics plot
            fig = self._create_metrics_plot()
            
            # Update drift alerts
            drift_html = self._create_drift_alerts()
            
            return metrics_html, fig, drift_html
    
    def _create_metrics_display(self):
        """Create current metrics display."""
        if not self.metrics_data['classification']['current'] and not self.metrics_data['regression']['current']:
            return html.P("Waiting for data...")
        
        cards = []
        
        # Classification metrics
        if self.metrics_data['classification']['current']:
            cm = self.metrics_data['classification']['current']
            cards.append(
                dbc.Card([
                    dbc.CardHeader("Classification Metrics"),
                    dbc.CardBody([
                        html.H5(f"Accuracy: {cm.get('cumulative', {}).get('accuracy', 0):.4f}"),
                        html.H5(f"F1 Score: {cm.get('cumulative', {}).get('f1', 0):.4f}"),
                        html.P(f"Instances: {cm.get('instance_count', 0)}")
                    ])
                ], className="m-2")
            )
        
        # Regression metrics
        if self.metrics_data['regression']['current']:
            rm = self.metrics_data['regression']['current']
            cards.append(
                dbc.Card([
                    dbc.CardHeader("Regression Metrics"),
                    dbc.CardBody([
                        html.H5(f"MAE: {rm.get('cumulative', {}).get('mae', 0):.4f}"),
                        html.H5(f"MSE: {rm.get('cumulative', {}).get('mse', 0):.4f}"),
                        html.H5(f"RMSE: {rm.get('cumulative', {}).get('rmse', 0):.4f}"),
                        html.P(f"Instances: {rm.get('instance_count', 0)}")
                    ])
                ], className="m-2")
            )
        
        return dbc.Row(cards)
    
    def _create_metrics_plot(self):
        """Create metrics plot."""
        fig = go.Figure()
        
        # Plot classification metrics
        if self.metrics_data['classification']['history']:
            hist = self.metrics_data['classification']['history']
            instances = [h['instance_count'] for h in hist if 'instance_count' in h]
            
            if instances:
                cum_acc = [h['cumulative_metrics'][-1]['accuracy'] 
                          for h in hist if 'cumulative_metrics' in h and h['cumulative_metrics']]
                cum_f1 = [h['cumulative_metrics'][-1]['f1'] 
                         for h in hist if 'cumulative_metrics' in h and h['cumulative_metrics']]
                
                if cum_acc:
                    fig.add_trace(go.Scatter(
                        x=instances[:len(cum_acc)],
                        y=cum_acc,
                        mode='lines',
                        name='Accuracy',
                        line=dict(color='blue')
                    ))
                
                if cum_f1:
                    fig.add_trace(go.Scatter(
                        x=instances[:len(cum_f1)],
                        y=cum_f1,
                        mode='lines',
                        name='F1 Score',
                        line=dict(color='green')
                    ))
        
        # Plot regression metrics
        if self.metrics_data['regression']['history']:
            hist = self.metrics_data['regression']['history']
            instances = [h['instance_count'] for h in hist if 'instance_count' in h]
            
            if instances:
                cum_mae = [h['cumulative_metrics'][-1]['mae'] 
                          for h in hist if 'cumulative_metrics' in h and h['cumulative_metrics']]
                
                if cum_mae:
                    fig.add_trace(go.Scatter(
                        x=instances[:len(cum_mae)],
                        y=cum_mae,
                        mode='lines',
                        name='MAE',
                        line=dict(color='red')
                    ))
        
        fig.update_layout(
            title="Performance Metrics Over Time",
            xaxis_title="Instance Count",
            yaxis_title="Metric Value",
            hovermode='x unified'
        )
        
        return fig
    
    def _create_drift_alerts(self):
        """Create drift alerts display."""
        if not self.drift_events:
            return html.P("No drift events detected yet.")
        
        alerts = []
        for event in self.drift_events[-10:]:  # Show last 10
            alerts.append(
                dbc.Alert(
                    f"Drift detected at instance {event.get('instance', 'N/A')} "
                    f"(Error: {event.get('error', 0):.4f})",
                    color="warning",
                    className="m-2"
                )
            )
        
        return html.Div(alerts)
    
    def update_metrics(self, metrics_data: Dict, task_type: str):
        """Update metrics data."""
        self.metrics_data[task_type]['current'] = metrics_data
        if 'history' in metrics_data:
            self.metrics_data[task_type]['history'].append(metrics_data)
    
    def update_drift_events(self, events: List[Dict]):
        """Update drift events."""
        self.drift_events = events
    
    def run(self, debug: bool = False):
        """Run the dashboard."""
        print(f"Starting dashboard on http://localhost:{self.port}")
        self.app.run_server(debug=debug, port=self.port, host='0.0.0.0')


def run_dashboard_standalone(port: int = 8050):
    """Run dashboard as standalone application."""
    dashboard = StreamingDashboard(port=port)
    dashboard.run(debug=True)

