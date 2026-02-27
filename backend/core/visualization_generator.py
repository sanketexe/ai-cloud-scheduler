"""
Visualization Generator for Natural Language Interface
Creates charts and data displays based on AI analysis results
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ChartType(str, Enum):
    """Supported chart types"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    AREA_CHART = "area_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    GAUGE_CHART = "gauge_chart"
    TREEMAP = "treemap"
    SANKEY_DIAGRAM = "sankey_diagram"
    ANOMALY_CHART = "anomaly_chart"


class ColorScheme(str, Enum):
    """Color schemes for visualizations"""
    BLUE_GRADIENT = "blue_gradient"
    GREEN_GRADIENT = "green_gradient"
    RED_GRADIENT = "red_gradient"
    RAINBOW = "rainbow"
    COST_OPTIMIZATION = "cost_optimization"
    ANOMALY_DETECTION = "anomaly_detection"


class VisualizationSpec:
    """Specification for a visualization"""
    
    def __init__(
        self,
        chart_type: ChartType,
        title: str,
        data: Any,
        config: Dict[str, Any] = None
    ):
        self.chart_type = chart_type
        self.title = title
        self.data = data
        self.config = config or {}
        self.created_at = datetime.now()


class VisualizationGenerator:
    """Generates visualizations for natural language interface responses"""
    
    def __init__(self):
        self.color_schemes = self._initialize_color_schemes()
        
    def _initialize_color_schemes(self) -> Dict[ColorScheme, List[str]]:
        """Initialize color schemes for different visualization types"""
        return {
            ColorScheme.BLUE_GRADIENT: [
                "#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5",
                "#2196F3", "#1E88E5", "#1976D2", "#1565C0", "#0D47A1"
            ],
            ColorScheme.GREEN_GRADIENT: [
                "#E8F5E8", "#C8E6C9", "#A5D6A7", "#81C784", "#66BB6A",
                "#4CAF50", "#43A047", "#388E3C", "#2E7D32", "#1B5E20"
            ],
            ColorScheme.RED_GRADIENT: [
                "#FFEBEE", "#FFCDD2", "#EF9A9A", "#E57373", "#EF5350",
                "#F44336", "#E53935", "#D32F2F", "#C62828", "#B71C1C"
            ],
            ColorScheme.RAINBOW: [
                "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
                "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
            ],
            ColorScheme.COST_OPTIMIZATION: [
                "#27AE60", "#F39C12", "#E74C3C", "#3498DB", "#9B59B6",
                "#1ABC9C", "#F1C40F", "#E67E22", "#95A5A6", "#34495E"
            ],
            ColorScheme.ANOMALY_DETECTION: [
                "#2ECC71", "#F39C12", "#E74C3C", "#3498DB", "#9B59B6"
            ]
        }
    
    def generate_cost_trend_chart(
        self,
        cost_data: List[Dict[str, Any]],
        title: str = "Cost Trend Over Time"
    ) -> VisualizationSpec:
        """Generate a line chart for cost trends"""
        
        try:
            # Process cost data
            processed_data = self._process_time_series_data(cost_data)
            
            config = {
                "type": "line",
                "data": {
                    "labels": processed_data["labels"],
                    "datasets": [{
                        "label": "Daily Cost",
                        "data": processed_data["values"],
                        "borderColor": self.color_schemes[ColorScheme.BLUE_GRADIENT][5],
                        "backgroundColor": self.color_schemes[ColorScheme.BLUE_GRADIENT][1],
                        "fill": True,
                        "tension": 0.4
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": title
                        },
                        "legend": {
                            "display": True,
                            "position": "top"
                        }
                    },
                    "scales": {
                        "x": {
                            "type": "time",
                            "time": {
                                "unit": "day"
                            },
                            "title": {
                                "display": True,
                                "text": "Date"
                            }
                        },
                        "y": {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": "Cost ($)"
                            },
                            "ticks": {
                                "callback": "function(value) { return '$' + value.toLocaleString(); }"
                            }
                        }
                    },
                    "interaction": {
                        "intersect": False,
                        "mode": "index"
                    }
                }
            }
            
            return VisualizationSpec(
                chart_type=ChartType.LINE_CHART,
                title=title,
                data=processed_data,
                config=config
            )
            
        except Exception as e:
            logger.error("Failed to generate cost trend chart", error=str(e))
            return self._generate_error_visualization("Cost Trend Chart", str(e))
    
    def generate_service_breakdown_chart(
        self,
        service_data: List[Dict[str, Any]],
        title: str = "Cost Breakdown by Service"
    ) -> VisualizationSpec:
        """Generate a pie chart for service cost breakdown"""
        
        try:
            # Process service data
            processed_data = self._process_categorical_data(service_data, "service", "cost")
            
            config = {
                "type": "pie",
                "data": {
                    "labels": processed_data["labels"],
                    "datasets": [{
                        "data": processed_data["values"],
                        "backgroundColor": self.color_schemes[ColorScheme.RAINBOW][:len(processed_data["labels"])],
                        "borderWidth": 2,
                        "borderColor": "#FFFFFF"
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": title
                        },
                        "legend": {
                            "display": True,
                            "position": "right"
                        },
                        "tooltip": {
                            "callbacks": {
                                "label": "function(context) { return context.label + ': $' + context.parsed.toLocaleString() + ' (' + Math.round(context.parsed / context.dataset.data.reduce((a, b) => a + b, 0) * 100) + '%)'; }"
                            }
                        }
                    }
                }
            }
            
            return VisualizationSpec(
                chart_type=ChartType.PIE_CHART,
                title=title,
                data=processed_data,
                config=config
            )
            
        except Exception as e:
            logger.error("Failed to generate service breakdown chart", error=str(e))
            return self._generate_error_visualization("Service Breakdown Chart", str(e))
    
    def generate_optimization_opportunities_chart(
        self,
        optimization_data: List[Dict[str, Any]],
        title: str = "Optimization Opportunities"
    ) -> VisualizationSpec:
        """Generate a bar chart for optimization opportunities"""
        
        try:
            # Process optimization data
            processed_data = self._process_categorical_data(
                optimization_data, "recommendation", "potential_savings"
            )
            
            # Sort by potential savings (descending)
            sorted_indices = sorted(
                range(len(processed_data["values"])),
                key=lambda i: processed_data["values"][i],
                reverse=True
            )
            
            sorted_labels = [processed_data["labels"][i] for i in sorted_indices]
            sorted_values = [processed_data["values"][i] for i in sorted_indices]
            
            config = {
                "type": "bar",
                "data": {
                    "labels": sorted_labels,
                    "datasets": [{
                        "label": "Potential Savings",
                        "data": sorted_values,
                        "backgroundColor": self.color_schemes[ColorScheme.COST_OPTIMIZATION][:len(sorted_labels)],
                        "borderColor": self.color_schemes[ColorScheme.GREEN_GRADIENT][6],
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": title
                        },
                        "legend": {
                            "display": False
                        }
                    },
                    "scales": {
                        "x": {
                            "title": {
                                "display": True,
                                "text": "Optimization Recommendations"
                            }
                        },
                        "y": {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": "Potential Savings ($)"
                            },
                            "ticks": {
                                "callback": "function(value) { return '$' + value.toLocaleString(); }"
                            }
                        }
                    }
                }
            }
            
            return VisualizationSpec(
                chart_type=ChartType.BAR_CHART,
                title=title,
                data={"labels": sorted_labels, "values": sorted_values},
                config=config
            )
            
        except Exception as e:
            logger.error("Failed to generate optimization opportunities chart", error=str(e))
            return self._generate_error_visualization("Optimization Opportunities Chart", str(e))
    
    def generate_anomaly_detection_chart(
        self,
        anomaly_data: List[Dict[str, Any]],
        title: str = "Cost Anomalies Detection"
    ) -> VisualizationSpec:
        """Generate a specialized chart for anomaly detection"""
        
        try:
            # Process anomaly data
            normal_data = []
            anomaly_points = []
            
            for point in anomaly_data:
                if point.get("is_anomaly", False):
                    anomaly_points.append({
                        "x": point["timestamp"],
                        "y": point["value"],
                        "confidence": point.get("confidence", 0.0)
                    })
                else:
                    normal_data.append({
                        "x": point["timestamp"],
                        "y": point["value"]
                    })
            
            config = {
                "type": "line",
                "data": {
                    "datasets": [
                        {
                            "label": "Normal Cost",
                            "data": normal_data,
                            "borderColor": self.color_schemes[ColorScheme.BLUE_GRADIENT][5],
                            "backgroundColor": self.color_schemes[ColorScheme.BLUE_GRADIENT][1],
                            "fill": False,
                            "pointRadius": 3
                        },
                        {
                            "label": "Anomalies",
                            "data": anomaly_points,
                            "borderColor": self.color_schemes[ColorScheme.RED_GRADIENT][5],
                            "backgroundColor": self.color_schemes[ColorScheme.RED_GRADIENT][5],
                            "fill": False,
                            "pointRadius": 8,
                            "pointHoverRadius": 10,
                            "showLine": False
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": title
                        },
                        "legend": {
                            "display": True,
                            "position": "top"
                        },
                        "tooltip": {
                            "callbacks": {
                                "afterLabel": "function(context) { if (context.datasetIndex === 1) { return 'Confidence: ' + Math.round(context.raw.confidence * 100) + '%'; } return ''; }"
                            }
                        }
                    },
                    "scales": {
                        "x": {
                            "type": "time",
                            "time": {
                                "unit": "day"
                            },
                            "title": {
                                "display": True,
                                "text": "Date"
                            }
                        },
                        "y": {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": "Cost ($)"
                            },
                            "ticks": {
                                "callback": "function(value) { return '$' + value.toLocaleString(); }"
                            }
                        }
                    }
                }
            }
            
            return VisualizationSpec(
                chart_type=ChartType.ANOMALY_CHART,
                title=title,
                data={"normal": normal_data, "anomalies": anomaly_points},
                config=config
            )
            
        except Exception as e:
            logger.error("Failed to generate anomaly detection chart", error=str(e))
            return self._generate_error_visualization("Anomaly Detection Chart", str(e))
    
    def generate_resource_utilization_heatmap(
        self,
        utilization_data: List[Dict[str, Any]],
        title: str = "Resource Utilization Heatmap"
    ) -> VisualizationSpec:
        """Generate a heatmap for resource utilization"""
        
        try:
            # Process utilization data into matrix format
            processed_data = self._process_heatmap_data(utilization_data)
            
            config = {
                "type": "heatmap",
                "data": processed_data,
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": title
                        },
                        "legend": {
                            "display": True,
                            "position": "right"
                        }
                    },
                    "scales": {
                        "x": {
                            "title": {
                                "display": True,
                                "text": "Time Period"
                            }
                        },
                        "y": {
                            "title": {
                                "display": True,
                                "text": "Resources"
                            }
                        }
                    }
                }
            }
            
            return VisualizationSpec(
                chart_type=ChartType.HEATMAP,
                title=title,
                data=processed_data,
                config=config
            )
            
        except Exception as e:
            logger.error("Failed to generate resource utilization heatmap", error=str(e))
            return self._generate_error_visualization("Resource Utilization Heatmap", str(e))
    
    def generate_forecast_chart(
        self,
        historical_data: List[Dict[str, Any]],
        forecast_data: List[Dict[str, Any]],
        title: str = "Cost Forecast"
    ) -> VisualizationSpec:
        """Generate a chart showing historical data and forecasts"""
        
        try:
            # Process historical and forecast data
            historical_processed = self._process_time_series_data(historical_data)
            forecast_processed = self._process_time_series_data(forecast_data)
            
            config = {
                "type": "line",
                "data": {
                    "labels": historical_processed["labels"] + forecast_processed["labels"],
                    "datasets": [
                        {
                            "label": "Historical Cost",
                            "data": historical_processed["values"] + [None] * len(forecast_processed["values"]),
                            "borderColor": self.color_schemes[ColorScheme.BLUE_GRADIENT][5],
                            "backgroundColor": self.color_schemes[ColorScheme.BLUE_GRADIENT][1],
                            "fill": False
                        },
                        {
                            "label": "Forecasted Cost",
                            "data": [None] * (len(historical_processed["values"]) - 1) + [historical_processed["values"][-1]] + forecast_processed["values"],
                            "borderColor": self.color_schemes[ColorScheme.GREEN_GRADIENT][5],
                            "backgroundColor": self.color_schemes[ColorScheme.GREEN_GRADIENT][1],
                            "borderDash": [5, 5],
                            "fill": False
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": title
                        },
                        "legend": {
                            "display": True,
                            "position": "top"
                        }
                    },
                    "scales": {
                        "x": {
                            "type": "time",
                            "time": {
                                "unit": "day"
                            },
                            "title": {
                                "display": True,
                                "text": "Date"
                            }
                        },
                        "y": {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": "Cost ($)"
                            },
                            "ticks": {
                                "callback": "function(value) { return '$' + value.toLocaleString(); }"
                            }
                        }
                    }
                }
            }
            
            return VisualizationSpec(
                chart_type=ChartType.LINE_CHART,
                title=title,
                data={
                    "historical": historical_processed,
                    "forecast": forecast_processed
                },
                config=config
            )
            
        except Exception as e:
            logger.error("Failed to generate forecast chart", error=str(e))
            return self._generate_error_visualization("Cost Forecast Chart", str(e))
    
    def _process_time_series_data(self, data: List[Dict[str, Any]]) -> Dict[str, List]:
        """Process time series data for chart consumption"""
        
        labels = []
        values = []
        
        for point in data:
            if "timestamp" in point and "value" in point:
                labels.append(point["timestamp"])
                values.append(float(point["value"]))
            elif "date" in point and "cost" in point:
                labels.append(point["date"])
                values.append(float(point["cost"]))
        
        return {"labels": labels, "values": values}
    
    def _process_categorical_data(
        self,
        data: List[Dict[str, Any]],
        label_key: str,
        value_key: str
    ) -> Dict[str, List]:
        """Process categorical data for chart consumption"""
        
        labels = []
        values = []
        
        for item in data:
            if label_key in item and value_key in item:
                labels.append(str(item[label_key]))
                values.append(float(item[value_key]))
        
        return {"labels": labels, "values": values}
    
    def _process_heatmap_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process data for heatmap visualization"""
        
        # Extract unique resources and time periods
        resources = set()
        time_periods = set()
        
        for point in data:
            if "resource" in point and "time_period" in point:
                resources.add(point["resource"])
                time_periods.add(point["time_period"])
        
        resources = sorted(list(resources))
        time_periods = sorted(list(time_periods))
        
        # Create matrix
        matrix = []
        for resource in resources:
            row = []
            for time_period in time_periods:
                # Find utilization value for this resource and time period
                value = 0
                for point in data:
                    if point.get("resource") == resource and point.get("time_period") == time_period:
                        value = point.get("utilization", 0)
                        break
                row.append(value)
            matrix.append(row)
        
        return {
            "matrix": matrix,
            "resources": resources,
            "time_periods": time_periods
        }
    
    def _generate_error_visualization(self, chart_name: str, error_message: str) -> VisualizationSpec:
        """Generate an error visualization when chart generation fails"""
        
        config = {
            "type": "error",
            "message": f"Failed to generate {chart_name}: {error_message}",
            "timestamp": datetime.now().isoformat()
        }
        
        return VisualizationSpec(
            chart_type=ChartType.BAR_CHART,  # Default fallback
            title=f"Error: {chart_name}",
            data={"error": True, "message": error_message},
            config=config
        )
    
    def generate_custom_visualization(
        self,
        data: Any,
        chart_type: ChartType,
        title: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> VisualizationSpec:
        """Generate a custom visualization with user-specified parameters"""
        
        try:
            base_config = {
                "type": chart_type.value.replace("_chart", ""),
                "data": data,
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": title
                        }
                    }
                }
            }
            
            # Merge custom configuration
            if custom_config:
                base_config = self._deep_merge_config(base_config, custom_config)
            
            return VisualizationSpec(
                chart_type=chart_type,
                title=title,
                data=data,
                config=base_config
            )
            
        except Exception as e:
            logger.error("Failed to generate custom visualization", error=str(e))
            return self._generate_error_visualization(f"Custom {chart_type.value}", str(e))
    
    def _deep_merge_config(self, base: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries"""
        
        result = base.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_config(result[key], value)
            else:
                result[key] = value
        
        return result


# Global instance
visualization_generator = VisualizationGenerator()