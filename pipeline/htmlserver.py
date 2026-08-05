import math
from dash import Dash, html, dcc, Input, Output, State, ALL, MATCH
import cv2
from flask import Flask, Response, jsonify, request
from pipeline.visionmanager import VisionManager
import time
from threading import Thread
from util.config import ConfigCategory
import os
class DashboardServer:
    config_category = ConfigCategory("HTMLServer")
    framecomp_slider = config_category.getFloatConfig("framecomp_slider", 0.5)

    def __init__(self, vision_manager: VisionManager, camera_id: int):
        self.vision_manager = vision_manager
        self.camera_id = camera_id
        self.server = Flask(__name__)
        self.app = Dash(__name__, server=self.server, suppress_callback_exceptions=True)
        self.app.index_string = self.index_string()
        
        self._calibrator = None
        
        self.setup_layout()
        self.setup_callbacks()
        self.start_server_thread()

    @property
    def calibrator(self):
        if self._calibrator is None:
            from calibration.checkerboard_calibrator import CheckerboardCalibrator
            self._calibrator = CheckerboardCalibrator(self.camera_id)
        return self._calibrator

    def create_detection_content(self, cam_id):
        pipeline = self.vision_manager.get_pipeline(cam_id)
        
        if not pipeline:
            return html.Div(f"Camera {cam_id} not available")
        
        if pipeline.get_pipeline_type() == "apriltag":
            settings_panel = self.create_apriltag_sliders(cam_id)
        else:
            settings_panel = self.create_gamepiece_sliders(cam_id)
        
        return html.Div([
                html.Div([
                    html.Div([
                        html.H4("Detections", style={
                            'textAlign': 'left',
                            'color': '#CCC9CA',
                            'font-size': '18px',
                            'font-weight': 'medium',
                            'padding': '0px 0px 0px 7px',
                        }),
                        html.Div(
                            id=f'detections-{cam_id}',
                            style={
                                'flex-direction': 'column',
                                'gap': '5px',
                                'margin-top': '5px',
                                'padding': '0 0px',
                            }
                        ),
                        html.Br(),
                        settings_panel,
                    ], style={
                        "flex": "1",
                        "padding": "10px",
                        "color": "#FFF",
                        "flex-direction": "column",
                        "display": "flex",
                        "flex-grow": "1",
                        "min-width": "40%",
                        "max-width": "50%",
                        "box-sizing": "border-box",
                        "overflow-y": "auto",
                        "overflow-x": "hidden",
                        "height": "85vh",
                    }),
                    
                    html.Div([
                        html.Div([
                            html.Div(f"{pipeline.get_camera_name()}", style={
                                'color': '#CDA646',
                                'font-size': '12px',
                                'font-weight': 'bold',
                                'margin-bottom': '5px',
                                'text-align': 'center',
                            }),
                            html.Img(
                                src=f"/video_feed/{cam_id}",
                                style={
                                    "width": "100%",
                                    "max-width": "650px",
                                    "max-height": "600px",
                                    "border": "3px solid #CDA646",
                                    "border-radius": "9px",
                                }
                            ),
                            html.Div(id=f'metrics-{cam_id}', style={
                                "position": "relative",
                                "width": "100%",
                                "max-width": "600px",
                                "height": "40px",
                                "margin-top": "5px",
                            }),
                            html.Br(),
                            html.Div([
                                html.Button("Reload", id={'type': 'reload', 'index': cam_id}, style={
                                    "margin": "10px 5px",
                                    "font-size": "14px",
                                    "color": "#161616",
                                    "background-color": "rgba(100, 200, 255, 1)",
                                    "border": "none",
                                    "padding": "8px 16px",
                                    "width": "145px",
                                    "height": "40px",
                                    "border-radius": "20px",
                                    "cursor": "pointer",
                                    "font-weight": "bold"
                                }),
                                html.Button("Reboot", id={'type': 'reboot', 'index': cam_id}, style={
                                    "margin": "10px 5px",
                                    "font-size": "14px",
                                    "color": "#161616",
                                    "background-color": "rgba(255, 204, 74, 1)",
                                    "border": "none",
                                    "padding": "8px 16px",
                                    "width": "145px",
                                    "height": "40px",
                                    "border-radius": "20px",
                                    "cursor": "pointer",
                                    "font-weight": "bold"
                                }),
                            ], style={
                                "display": "flex",
                                "justify-content": "center",
                                "align-items": "center",
                                "padding": "10px",
                            }),
                        ], style={
                            "display": "flex",
                            "flex-direction": "column",
                            "align-items": "center",
                            "justify-content": "center",
                        }),
                    ], style={
                        "flex": "3",
                        "display": "flex",
                        "justify-content": "center",
                        "align-items": "center",
                        "padding": "20px",
                        "height": "85vh",
                    })
                ], style={
                    "display": "flex",
                    "flex-direction": "row",
                    "width": "100%",
                    "padding": "10px 10px",
                    "overflow-x": "hidden",
                })
            ])

    def setup_layout(self):
        camera_content = self.create_detection_content(self.camera_id)
        
        self.app.layout = html.Div([
            html.Div([
                html.H1("MonkeyVision", style={
                    'textAlign': 'left',
                    'color': '#CCC9CA',
                    'font-size': '32px',
                    'font-weight': 'bold',
                    'padding': '10px 20px 0 25px',
                    'margin-bottom': '5px',
                }),
                html.H4("By Team 846 • The Funky Monkeys", style={
                    'textAlign': 'left',
                    'color': '#CCC9CA',
                    'font-size': '14px',
                    'font-weight': 'regular',
                    'margin': '5px 0 10px 25px',
                }),
                html.Img(
                    src="/assets/logo.svg",
                    style={
                        "position": "absolute",
                        "top": "20px",
                        "right": "20px",
                        "width": "50px",
                        "height": "50px"
                    }
                ),
            ]),
            
            dcc.Tabs(id='tabs', value='detection', children=[
                dcc.Tab(label='Detection', value='detection', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Calibration', value='calibration', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Focus', value='focus', className='custom-tab', selected_className='custom-tab--selected'),
            ], style={
                'margin': '0 25px',
                'border-bottom': '2px solid #CDA646',
            }),
            
            html.Div(id='tab-content', children=camera_content),
            
            dcc.Interval(id="update-interval", interval=500, n_intervals=0),  # 500ms for faster progress updates
        ], style={
            "background-color": "#161616",
            "color": "#FFF",
            "font-family": "'Inter', sans-serif",
            "min-height": "100vh",
            "padding": "0",
            "margin": "0",
        })

    def create_apriltag_sliders(self, cam_id):
        from camera.preprocess import GET_DIVERGENCE_GAIN, GET_TARGET_BRIGHTNESS, GET_NUM_BINS, GET_MIN_CORR_STRENGTH
        from localization.detection import GET_THRESH_STEP, GET_THRESH_WIN
        
        return html.Div([
            html.H4("Settings", style={
                'textAlign': 'left',
                'color': '#CCC9CA',
                'font-size': '18px',
                'font-weight': 'medium',
                'padding': '15px 0px 0px 7px',
            }),
            
            html.Label("Dynamic Frame Correction Target", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                "margin-bottom": "10px",
                'padding': '15px 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'dfc-target', 'index': cam_id},
                min=50,
                max=200,
                step=1,
                value=GET_TARGET_BRIGHTNESS(),
                marks={50: '50', 200: '200'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("ASLC Num Bins", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'aslc-bins', 'index': cam_id},
                min=100,
                max=1600,
                step=100,
                value=GET_NUM_BINS(),
                marks={100: '100', 1600: '1600'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("Min ASLC Correction", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'aslc-min-corr', 'index': cam_id},
                min=0.02,
                max=0.5,
                step=0.01,
                value=GET_MIN_CORR_STRENGTH(),
                marks={0.02: '0.02', 0.5: '0.5'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("Divergence Gain", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'divergence-gain', 'index': cam_id},
                min=0.5,
                max=4.0,
                step=0.1,
                value=GET_DIVERGENCE_GAIN(),
                marks={0.5: '0.5', 4.0: '4.0'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("Thresholding Steps", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'thresh-step', 'index': cam_id},
                min=11,
                max=33,
                step=2,
                value=GET_THRESH_STEP(),
                marks={11: '11', 33: '33'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("Maximum Threshold Window", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'max-thresh', 'index': cam_id},
                min=11,
                max=33,
                step=2,
                value=GET_THRESH_WIN(),
                marks={11: '11', 33: '33'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),
            
            html.Label("Displayed Frame Quality", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'framecomp', 'index': cam_id},
                min=0.05,
                max=1,
                step=0.05,
                value=DashboardServer.framecomp_slider.valueFloat(),
                marks={0.05: '5%', 1: '100%'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
        ])

    def create_calibration_tab(self, cam_id):
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Camera Calibration", style={
                        'textAlign': 'left',
                        'color': '#CCC9CA',
                        'font-size': '18px',
                        'font-weight': 'medium',
                        'padding': '0px 0px 0px 7px',
                    }),
                    
                    html.Div([
                        html.P("https://calib.io/pages/camera-calibration-pattern-generator", style={'margin': '10px 0'}),
                        html.P("11 rows, 8 columns", style={'margin': '10px 0'}),
                        html.P("15mm ChArUco checkers. Class 4x4.", style={'margin': '10px 0'}),
                        html.P("Printed to 21.2mm checkers, 15.55mm markers", style={'margin': '10px 0'}),
                    ], style={
                        'color': '#CCC9CA',
                        'font-size': '14px',
                        'padding': '15px 20px',
                        'border': '2px solid rgba(255, 255, 255, 0.3)',
                        'border-radius': '10px',
                        'margin': '15px 7px',
                    }),
                    
                    html.Div(id=f'calibration-status-{cam_id}', style={
                        'padding': '15px 20px',
                        'margin': '15px 7px',
                        'border': '2px solid rgba(255, 255, 255, 0.3)',
                        'border-radius': '10px',
                        'color': '#CCC9CA',
                        'font-size': '14px',
                    }),
                    
                    html.Div([
                        html.Button("Calibrate", id={'type': 'run-calibration', 'index': cam_id}, style={
                            "margin": "10px 5px",
                            "font-size": "14px",
                            "color": "#161616",
                            "background-color": "rgba(100, 200, 100, 1)",
                            "border": "none",
                            "padding": "8px 16px",
                            "width": "180px",
                            "height": "40px",
                            "border-radius": "20px",
                            "cursor": "pointer",
                            "font-weight": "bold"
                        }),
                        html.Button("Reset", id={'type': 'reset-calibration', 'index': cam_id}, style={
                            "margin": "10px 5px",
                            "font-size": "14px",
                            "color": "#161616",
                            "background-color": "rgba(255, 100, 100, 1)",
                            "border": "none",
                            "padding": "8px 16px",
                            "width": "180px",
                            "height": "40px",
                            "border-radius": "20px",
                            "cursor": "pointer",
                            "font-weight": "bold"
                        }),
                    ], style={
                        "display": "flex",
                        "flex-direction": "column",
                        "align-items": "center",
                        "padding": "10px",
                    }),
                    
                ], style={
                    "flex": "1",
                    "padding": "10px",
                    "color": "#FFF",
                    "flex-direction": "column",
                    "display": "flex",
                    "flex-grow": "1",
                    "min-width": "40%",
                    "max-width": "50%",
                    "box-sizing": "border-box",
                    "overflow-y": "auto",
                    "overflow-x": "hidden",
                    "height": "85vh",
                }),
                
                html.Div([
                    html.Div([
                        html.Div(f"Calibration Preview - Camera {cam_id}", style={
                            'color': '#CDA646',
                            'font-size': '12px',
                            'font-weight': 'bold',
                            'margin-bottom': '5px',
                            'text-align': 'center',
                        }),
                        html.Img(
                            src=f"/calibration_feed/{cam_id}",
                            style={
                                "width": "100%",
                                "max-width": "650px",
                                "max-height": "600px",
                                "border": "3px solid #CDA646",
                            "border-radius": "9px",
                        }
                    ),
                ], style={
                    "display": "flex",
                    "flex-direction": "column",
                    "align-items": "center",
                    "justify-content": "center",
                }),
            ], style={
                "flex": "3",
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
                "padding": "20px",
                "height": "85vh",
            })
        ], style={
            "display": "flex",
            "flex-direction": "row",
            "width": "100%",
            "padding": "10px 10px",
            "overflow-x": "hidden",
        }),
        
        html.Div(id=f'calibration-results-{cam_id}', style={
            "width": "100%",
            "padding": "20px",
            "display": "none",  # Hidden by default
        })
    ])

    def create_focus_tab(self, cam_id):
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Camera Focusing", style={
                        'textAlign': 'left',
                        'color': '#CCC9CA',
                        'font-size': '18px',
                        'font-weight': 'medium',
                        'padding': '0px 0px 0px 7px',
                    }),
                    html.P("Rotate the lens to maximize the score. Ensure the camera is pointed at a highly textured surface (like an AprilTag or calibration board).", style={
                        'color': '#CCC9CA',
                        'font-size': '14px',
                        'padding': '15px 20px',
                        'margin': '15px 7px',
                    }),
                    html.Div([
                        html.Button("Enable Focus Mode", id={'type': 'toggle-focus', 'index': cam_id}, style={
                            "margin": "10px 5px",
                            "font-size": "14px",
                            "color": "#161616",
                            "background-color": "rgba(255, 204, 74, 1)",
                            "border": "none",
                            "padding": "8px 16px",
                            "width": "200px",
                            "height": "40px",
                            "border-radius": "20px",
                            "cursor": "pointer",
                            "font-weight": "bold"
                        }),
                    ], style={
                        "display": "flex",
                        "justify-content": "center",
                        "align-items": "center",
                        "padding": "10px",
                    }),
                    html.Div(id=f'focus-score-{cam_id}', children="Score: 0.0", style={
                        'color': '#00ff00',
                        'font-size': '48px',
                        'font-weight': 'bold',
                        'text-align': 'center',
                        'padding': '20px',
                        'margin-top': '20px'
                    }),
                ], style={
                    "flex": "1",
                    "padding": "10px",
                    "color": "#FFF",
                    "flex-direction": "column",
                    "display": "flex",
                    "flex-grow": "1",
                    "min-width": "40%",
                    "max-width": "50%",
                    "box-sizing": "border-box",
                    "overflow-y": "auto",
                    "overflow-x": "hidden",
                    "height": "85vh",
                }),
                html.Div([
                    html.Div([
                        html.Div(f"Focus Preview - Camera {cam_id}", style={
                            'color': '#CDA646',
                            'font-size': '12px',
                            'font-weight': 'bold',
                            'margin-bottom': '5px',
                            'text-align': 'center',
                        }),
                        html.Img(
                            src=f"/video_feed/{cam_id}",
                            style={
                                "width": "100%",
                                "max-width": "650px",
                                "max-height": "600px",
                                "border": "3px solid #CDA646",
                                "border-radius": "9px",
                            }
                        ),
                    ], style={
                        "display": "flex",
                        "flex-direction": "column",
                        "align-items": "center",
                        "justify-content": "center",
                    }),
                ], style={
                    "flex": "3",
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                    "padding": "20px",
                    "height": "85vh",
                })
            ], style={
                "display": "flex",
                "flex-direction": "row",
                "width": "100%",
                "padding": "10px 10px",
                "overflow-x": "hidden",
            }),
        ])

    def create_gamepiece_sliders(self, cam_id):
        from localization.visiony import CONF, ASPECT_THRESH
        from localization.gamepiece_solution import WD, ONT
        
        return html.Div([
            html.H4("YOLO Settings", style={
                'textAlign': 'left',
                'color': '#CCC9CA',
                'font-size': '18px',
                'font-weight': 'medium',
                'padding': '15px 0px 0px 7px',
            }),
            
            html.Label("Detection Confidence", style={"color": "#CCC9CA", "font-size": "16px", 'padding': '15px 20px 0 15px'}),
            dcc.Slider(id={'type': 'yolo_conf', 'index': cam_id}, min=0.1, max=0.95, step=0.05, value=CONF.valueFloat(),
                      marks={0.1: '0.1', 0.5: '0.5', 0.95: '0.95'}, tooltip={"placement": "bottom", "always_visible": True}, className="funky-slider"),
            html.Br(),
            
            html.Label("Aspect Ratio Threshold", style={"color": "#CCC9CA", "font-size": "16px", 'padding': '0 20px 0 15px'}),
            dcc.Slider(id={'type': 'yolo_aspect', 'index': cam_id}, min=0.0, max=1.0, step=0.05, value=ASPECT_THRESH.valueFloat(),
                      marks={0.0: '0.0', 0.5: '0.5', 1.0: '1.0'}, tooltip={"placement": "bottom", "always_visible": True}, className="funky-slider"),
            html.Br(),
            
            html.H4("Distance Settings", style={
                'textAlign': 'left',
                'color': '#CCC9CA',
                'font-size': '18px',
                'font-weight': 'medium',
                'padding': '15px 0px 0px 7px',
            }),
            
            html.Label("Width Distance (WD)", style={"color": "#CCC9CA", "font-size": "16px", 'padding': '15px 20px 0 15px'}),
            dcc.Slider(id={'type': 'wd', 'index': cam_id}, min=5.0, max=30.0, step=0.5, value=WD.valueFloat(),
                      marks={5.0: '5.0', 15.5: '15.5', 30.0: '30.0'}, tooltip={"placement": "bottom", "always_visible": True}, className="funky-slider"),
            html.Br(),
            
            html.Label("On Top Threshold (ONT)", style={"color": "#CCC9CA", "font-size": "16px", 'padding': '0 20px 0 15px'}),
            dcc.Slider(id={'type': 'ont', 'index': cam_id}, min=0.0, max=20.0, step=1.0, value=ONT.valueFloat(),
                      marks={0.0: '0.0', 10.0: '10.0', 20.0: '20.0'}, tooltip={"placement": "bottom", "always_visible": True}, className="funky-slider"),
            html.Br(),
            
            html.Label("Displayed Frame Quality", style={"color": "#CCC9CA", "font-size": "16px", 'padding': '0 20px 0 15px'}),
            dcc.Slider(id={'type': 'framecomp', 'index': cam_id}, min=0.05, max=1, step=0.05,
                      value=DashboardServer.framecomp_slider.valueFloat(),
                      marks={0.05: '5%', 1: '100%'}, tooltip={"placement": "bottom", "always_visible": True}, className="funky-slider"),
        ])

    def setup_callbacks(self):
        from dash import no_update
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('tabs', 'value')]
        )
        def render_tab_content(tab):
            if tab == 'calibration':
                return self.create_calibration_tab(self.camera_id)
            elif tab == 'focus':
                return self.create_focus_tab(self.camera_id)
            else:  # detection tab
                return self.create_detection_content(self.camera_id)
        
        cam_id = self.camera_id
        
        @self.app.callback(
            Output(f'metrics-{cam_id}', 'children'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_metrics(n_intervals, camera_id=cam_id):
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if pipeline and pipeline.is_enabled():
                backend_fps = pipeline.get_backend_fps()
                fps_display = f"{backend_fps:.1f}" if backend_fps > 0 else f"{pipeline.get_framerate():.1f}"
                return [
                    html.Span(f"Processing: {fps_display} FPS", style={
                        "position": "absolute",
                        "left": "0",
                        "bottom": "0",
                        "color": "rgba(255, 255, 255, 0.8)",
                        "font-size": "18px",
                        "font-weight": "regular",
                        "font-style": "italic",
                        "padding": "5px 10px",
                    }),
                    html.Span(f"Latency: {pipeline.get_processing_latency() * 1000:.2f} ms", style={
                        "position": "absolute",
                        "right": "0",
                        "bottom": "0",
                        "color": "rgba(255, 255, 255, 0.8)",
                        "font-size": "18px",
                        "font-weight": "regular",
                        "padding": "5px 10px",
                        "font-style": "italic",
                        "border-radius": "5px",
                    })
                ]
            return [html.Span("Disabled", style={'color': '#888'})]
        
        @self.app.callback(
            Output(f'detections-{cam_id}', 'children'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_detections(n_intervals, camera_id=cam_id):
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if pipeline and pipeline.is_enabled():
                detections = pipeline.get_detections()
                if not detections:
                    return [html.Div("No detections", style={
                        'color': '#CCC9CA',
                        'font-size': '14px',
                        'text-align': 'center',
                        'border-radius': '10px',
                        'border': '2px solid rgba(255, 255, 255, 0.5)',
                        'padding': '10px',
                        'margin': '0 0px 20px 20px',
                    })]
                
                detection_items = []
                for i, det in enumerate(detections):
                    if pipeline.get_pipeline_type() == "apriltag":
                        detection_items.append(html.Div([
                            html.Span(f"Detection #{i + 1}:", style={'color': '#CCC9CA', 'margin-right': '10px', 'font-weight': 'medium'}),
                            html.Span(f"Tag {det.getTag()}", style={'color': '#CCC9CA', 'margin-right': '5px'}),
                            html.Span(f"R {det.getR():.1f}in", style={'color': '#CCC9CA', 'margin-right': '5px'}),
                            html.Span(f"θ {det.getTheta():.2f}deg", style={'color': '#CCC9CA'}),
                        ], style={
                            'border': '2px solid rgba(255, 255, 255, 0.5)',
                            'border-radius': '10px',
                            'padding': '10px',
                            'font-size': '14px',
                            'margin': '0 0px 20px 20px',
                            'width': '100%',
                        }))
                    else:
                        detection_items.append(html.Div([
                            html.Span(f"Detection #{i + 1}:", style={'color': '#CCC9CA', 'margin-right': '10px', 'font-weight': 'medium'}),
                            html.Span(f"R {det.getR():.2f}in", style={'color': '#CCC9CA', 'margin-right': '5px'}),
                            html.Span(f"θ {det.getTheta():.2f}deg", style={'color': '#CCC9CA', 'margin-right': '10px'}),
                            html.Span(f"On: {det.isOnTop()}", style={'color': '#CCC9CA'}),
                        ], style={
                            'border': '2px solid rgba(255, 255, 255, 0.5)',
                            'border-radius': '10px',
                            'padding': '10px',
                            'font-size': '14px',
                            'margin': '0 0px 20px 20px',
                            'width': '100%',
                        }))
                return detection_items
            return [html.Div("Camera disabled", style={'color': '#888'})]
    
        self.setup_slider_callbacks()
        
        self.server.add_url_rule(f'/video_feed/{cam_id}', f'video_feed_{cam_id}',
                                lambda cid=cam_id: self.video_feed(cid))
        self.server.add_url_rule(f'/calibration_feed/{cam_id}', f'calibration_feed_{cam_id}',
                                lambda cid=cam_id: self.calibration_feed(cid))

    def setup_slider_callbacks(self):
        from dash import no_update
        from camera.preprocess import SET_DIVERGENCE_GAIN, SET_TARGET_BRIGHTNESS, SET_NUM_BINS, SET_MIN_CORR_STRENGTH
        from localization.detection import SET_THRESH_STEP, SET_THRESH_WIN
        from localization.visiony import CONF, ASPECT_THRESH
        from localization.gamepiece_solution import WD, ONT
        
        @self.app.callback(
            Output({'type': 'dfc-target', 'index': MATCH}, 'value'),
            [Input({'type': 'dfc-target', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_dfc_target(value):
            if value is not None:
                SET_TARGET_BRIGHTNESS(value)
            return value
        
        @self.app.callback(
            Output({'type': 'aslc-bins', 'index': MATCH}, 'value'),
            [Input({'type': 'aslc-bins', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_aslc_bins(value):
            if value is not None:
                SET_NUM_BINS(value)
            return value
        
        @self.app.callback(
            Output({'type': 'aslc-min-corr', 'index': MATCH}, 'value'),
            [Input({'type': 'aslc-min-corr', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_aslc_min_corr(value):
            if value is not None:
                SET_MIN_CORR_STRENGTH(value)
            return value
        
        @self.app.callback(
            Output({'type': 'divergence-gain', 'index': MATCH}, 'value'),
            [Input({'type': 'divergence-gain', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_divergence_gain(value):
            if value is not None:
                SET_DIVERGENCE_GAIN(value)
            return value
        
        @self.app.callback(
            Output({'type': 'thresh-step', 'index': MATCH}, 'value'),
            [Input({'type': 'thresh-step', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_thresh_step(value):
            if value is not None:
                SET_THRESH_STEP(value)
            return value
        
        @self.app.callback(
            Output({'type': 'max-thresh', 'index': MATCH}, 'value'),
            [Input({'type': 'max-thresh', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_max_thresh(value):
            if value is not None:
                SET_THRESH_WIN(value)
            return value
        
        @self.app.callback(
            Output({'type': 'framecomp', 'index': MATCH}, 'value'),
            [Input({'type': 'framecomp', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_framecomp(value):
            if value is not None:
                DashboardServer.framecomp_slider.setFloat(value)
            return value
        
        @self.app.callback(
            Output({'type': 'yolo_conf', 'index': MATCH}, 'value'),
            [Input({'type': 'yolo_conf', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_yolo_conf(value):
            if value is not None:
                CONF.setFloat(value)
            return value
        
        @self.app.callback(
            Output({'type': 'yolo_aspect', 'index': MATCH}, 'value'),
            [Input({'type': 'yolo_aspect', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_yolo_aspect(value):
            if value is not None:
                ASPECT_THRESH.setFloat(value)
            return value
        
        @self.app.callback(
            Output({'type': 'wd', 'index': MATCH}, 'value'),
            [Input({'type': 'wd', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_wd(value):
            if value is not None:
                WD.setFloat(value)
            return value
        
        @self.app.callback(
            Output({'type': 'ont', 'index': MATCH}, 'value'),
            [Input({'type': 'ont', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_ont(value):
            if value is not None:
                ONT.setFloat(value)
            return value
        
        @self.app.callback(
            Output({'type': 'reboot', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'reboot', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def reboot_system(n_clicks):
            if n_clicks:
                os.system('sudo reboot')
            return n_clicks
        
        @self.app.callback(
            Output({'type': 'reload', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'reload', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def reload_pipeline(n_clicks):
            if n_clicks:
                Thread(target=self.vision_manager.reload_all_pipelines, daemon=True).start()
            return n_clicks
        
        self.setup_calibration_callbacks()
        self.setup_focus_callbacks()

    def setup_focus_callbacks(self):
        cam_id = self.camera_id
        
        @self.app.callback(
            Output(f'focus-score-{cam_id}', 'children'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_focus_score(n_intervals):
            pipeline = self.vision_manager.get_pipeline(cam_id)
            if pipeline and pipeline.focus_mode:
                return f"Score: {pipeline.focus_score:.1f}"
            return "Focus Mode Disabled"
            
        @self.app.callback(
            Output({'type': 'toggle-focus', 'index': MATCH}, 'children'),
            [Input({'type': 'toggle-focus', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def toggle_focus_mode(n_clicks):
            if n_clicks:
                pipeline = self.vision_manager.get_pipeline(cam_id)
                if pipeline:
                    pipeline.focus_mode = not pipeline.focus_mode
                    return "Disable Focus Mode" if pipeline.focus_mode else "Enable Focus Mode"
            from dash import no_update
            return no_update

    def setup_calibration_callbacks(self):
        cam_id = self.camera_id
        
        @self.app.callback(
            Output(f'calibration-status-{cam_id}', 'children'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_calibration_status(n_intervals):
            status = self.calibrator.get_status()
            progress_pct = status['progress'] * 100
            progress_color = '#00ff00' if status['is_calibrating'] else '#ffaa00'
            
            # Show progress bar during calibration
            progress_bar = None
            if status['is_calibrating']:
                progress_bar = html.Div([
                    html.Div(style={
                        'width': f'{progress_pct:.1f}%',
                        'height': '20px',
                        'background-color': '#00ff00',
                        'transition': 'width 0.3s ease',
                        'border-radius': '4px',
                    }),
                ], style={
                    'width': '100%',
                    'height': '20px',
                    'background-color': '#333',
                    'border-radius': '4px',
                    'overflow': 'hidden',
                    'margin': '5px 0',
                })
            
            return html.Div([
                html.Div(f"Status: {status['status_message']}", style={'margin': '5px 0'}),
                html.Div(f"Samples: {status['num_samples']}/{status['target_samples']}", style={'margin': '5px 0'}),
                html.Div(f"Progress: {progress_pct:.1f}%", style={'margin': '5px 0'}),
                progress_bar if progress_bar else html.Div(),
                html.Div(
                    "✓ Ready to calibrate" if status['is_ready'] and not status['is_calibrating'] else 
                    ("Calibrating..." if status['is_calibrating'] else "Need more samples"),
                    style={'margin': '5px 0', 'color': progress_color}
                ),
            ])
        
        @self.app.callback(
            Output(f'calibration-results-{cam_id}', 'children'),
            Output(f'calibration-results-{cam_id}', 'style'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_calibration_results(n_intervals):
            result = self.calibrator.calibration_result
            if result is None:
                return [], {'display': 'none'}
            
            meta = result.get('meta', {})
            fov = meta.get('fov', {})
            distortion_viz = meta.get('distortion_viz_base64', '')
            
            return [
                html.H4("Calibration Results", style={
                    'color': '#CDA646',
                    'margin': '20px 0 10px 0',
                    'text-align': 'center',
                }),
                html.Div([
                    html.Div([
                        html.H5("Field of View", style={'color': '#CCC9CA', 'margin': '10px 0'}),
                        html.Div(f"Horizontal: {fov.get('horizontal', 0):.1f}°", 
                                style={'color': '#CCC9CA', 'margin': '5px 0'}),
                        html.Div(f"Vertical: {fov.get('vertical', 0):.1f}°", 
                                style={'color': '#CCC9CA', 'margin': '5px 0'}),
                        html.Div(f"Diagonal: {fov.get('diagonal', 0):.1f}°", 
                                style={'color': '#CCC9CA', 'margin': '5px 0'}),
                        html.Div(f"Reprojection Error: {meta.get('reprojection_error', 0):.4f} px", 
                                style={'color': '#CCC9CA', 'margin': '15px 0 5px 0'}),
                        html.Div(f"Samples Used: {meta.get('num_samples', 0)}", 
                                style={'color': '#CCC9CA', 'margin': '5px 0'}),
                    ], style={
                        'flex': '1',
                        'padding': '20px',
                        'border': '2px solid #CDA646',
                        'border-radius': '10px',
                        'margin': '10px',
                    }),
                    html.Div([
                        html.Img(src=f"data:image/png;base64,{distortion_viz}",
                                style={
                                    'max-width': '100%',
                                    'border': '2px solid #CDA646',
                                    'border-radius': '5px',
                                }),
                    ], style={
                        'flex': '2',
                        'padding': '20px',
                        'margin': '10px',
                    }),
                ], style={
                    'display': 'flex',
                    'flex-direction': 'row',
                    'align-items': 'center',
                    'justify-content': 'center',
                }),
                html.Div([
                    html.Button("Save Calibration", id={'type': 'save-calibration', 'index': cam_id}, style={
                        "margin": "20px 5px",
                        "font-size": "16px",
                        "color": "#161616",
                        "background-color": "rgba(100, 200, 100, 1)",
                        "border": "none",
                        "padding": "12px 24px",
                        "width": "250px",
                        "height": "50px",
                        "border-radius": "25px",
                        "cursor": "pointer",
                        "font-weight": "bold"
                    }),
                ], style={
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                })
            ], {
                'width': '100%',
                'padding': '20px',
                'display': 'block',
                'background-color': 'rgba(0, 0, 0, 0.3)',
                'border-top': '2px solid #CDA646',
            }
        
        @self.app.callback(
            Output({'type': 'run-calibration', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'run-calibration', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def run_calibration(n_clicks):
            if n_clicks:
                Thread(target=self.calibrator.calibrate, daemon=True).start()
            return n_clicks
        
        @self.app.callback(
            Output({'type': 'reset-calibration', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'reset-calibration', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def reset_calibration(n_clicks):
            if n_clicks:
                self.calibrator.reset()
            return n_clicks
        
        @self.app.callback(
            Output({'type': 'save-calibration', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'save-calibration', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def save_calibration(n_clicks):
            if n_clicks:
                self.calibrator.save_calibration()
            return n_clicks
        
    def video_feed(self, camera_id: int):
        return Response(self.generate_frames(camera_id),
                       mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_frames(self, camera_id: int):
        blank_frame_cache = None
        last_frame = None
        frame_skip_counter = 0
        
        while True:
            pipeline = self.vision_manager.get_pipeline(camera_id)
            
            if not pipeline or not pipeline.is_enabled():
                if blank_frame_cache is None:
                    blank = 255 * cv2.ones((480, 640, 3), dtype='uint8')
                    cv2.putText(blank, "Camera Disabled", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                    encode_params = [
                        cv2.IMWRITE_JPEG_QUALITY, 60,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                        cv2.IMWRITE_JPEG_PROGRESSIVE, 0
                    ]
                    ret, buffer = cv2.imencode('.jpg', blank, encode_params)
                    blank_frame_cache = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + blank_frame_cache + b'\r\n')
                time.sleep(0.1)
                continue
            
            frame = pipeline.get_frame()
            if frame is None:
                if last_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')
                time.sleep(0.05)
                continue
            
            frame_skip_counter += 1
            if frame_skip_counter % 2 != 0:
                if last_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')
                continue
            
            quality = int(DashboardServer.framecomp_slider.valueFloat() * 100)
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, quality,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                cv2.IMWRITE_JPEG_PROGRESSIVE, 0
            ]
            
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame_bytes = buffer.tobytes()
            last_frame = frame_bytes
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def calibration_feed(self, camera_id: int):
        return Response(self.generate_calibration_frames(camera_id),
                       mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_calibration_frames(self, camera_id: int):
        blank_frame_cache = None
        last_frame = None
        
        while True:
            pipeline = self.vision_manager.get_pipeline(camera_id)
            
            if not pipeline:
                if blank_frame_cache is None:
                    blank = 255 * cv2.ones((480, 640, 3), dtype='uint8')
                    cv2.putText(blank, "Camera Not Available", (180, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                    ret, buffer = cv2.imencode('.jpg', blank)
                    blank_frame_cache = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + blank_frame_cache + b'\r\n')
                time.sleep(0.1)
                continue
            
            frame = pipeline.cam.get_raw_frame()
            
            if frame is None:
                if last_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')
                time.sleep(0.05)
                continue
            
            try:
                annotated_frame, is_good_sample, status = self.calibrator.process_frame(frame, auto_capture=True)
            except Exception as e:
                import traceback
                from util.logger import Logger
                logger = Logger("DashboardServer")
                logger.Error(f"Error processing calibration frame: {e}")
                logger.Error(traceback.format_exc())
                if last_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')
                time.sleep(0.05)
                continue
            
            if self.calibrator.try_auto_capture():
                cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), 
                             (0, 255, 0), 10)
                cv2.putText(annotated_frame, "CAPTURED!", (annotated_frame.shape[1]//2 - 100, annotated_frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            if annotated_frame is None:
                if last_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')
                time.sleep(0.05)
                continue
            
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, 70,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            ]
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
            frame_bytes = buffer.tobytes()
            last_frame = frame_bytes
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)  # ~30 FPS

    def start_server(self):
        port = 5800 + self.camera_id
        print(f"\n{'='*50}")
        print(f"MonkeySee Dashboard Starting for Camera {self.camera_id}")
        print(f"Port: {port}")
        print(f"Access at: http://0.0.0.0:{port}")
        print(f"{'='*50}\n")
        self.app.run_server(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    def start_server_thread(self):
        Thread(target=self.start_server, daemon=True).start()

    def index_string(self):
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MonkeyVision</title>
            <style>
                body {
                    background-color: #161616;
                    color: white;
                    font-family: 'Inter', sans-serif;
                    font-size: 15px;
                    margin: 0;
                    padding: 0;
                }
                .funky-slider .rc-slider-track {
                    background-color: #CDA646;
                }
                .funky-slider .rc-slider-rail {
                    background-color: rgba(255, 255, 255, 0.5);
                }
                .funky-slider .rc-slider-handle {
                    border-color: #CDA646;
                    background-color: #CDA646;
                }
                .funky-slider .rc-slider-tooltip {
                    font-size: 14px;
                    color: rgba(255, 255, 255, 0.5);
                    background-color: #CDA646;
                    border-radius: 8px;
                    box-shadow: none;
                }
                .custom-tab {
                    background-color: #161616 !important;
                    color: #CCC9CA !important;
                    border: none !important;
                    padding: 12px 24px !important;
                    font-size: 16px !important;
                    font-weight: 500 !important;
                    border-bottom: 3px solid transparent !important;
                }
                .custom-tab:hover {
                    background-color: #2a2a2a !important;
                }
                .custom-tab--selected {
                    background-color: #161616 !important;
                    color: #CDA646 !important;
                    border-bottom: 3px solid #CDA646 !important;
                }
                ::-webkit-scrollbar {
                    width: 10px;
                }
                ::-webkit-scrollbar-track {
                    background: transparent;
                }
                ::-webkit-scrollbar-thumb {
                    background: #CDA646;
                    border-radius: 5px;
                }
                ::-webkit-scrollbar-thumb:hover {
                    background: #b8923d;
                }
                * {
                    scrollbar-width: thin;
                    scrollbar-color: #CDA646 transparent;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            {%config%}
            {%scripts%}
            {%renderer%}
        </body>
        </html>
        """
