import math
from dash import Dash, html, dcc, Input, Output, State, ALL, MATCH
import cv2
from flask import Flask, Response, jsonify, request
from pipeline.visionmanager import VisionManager
import time
from threading import Thread
from util.config import ConfigCategory
from util.system_stats import get_system_stats
from camera.camerareader import RESOLUTIONS
from localization.undistort import has_opencv_calibration
from calibration.checkerboard_calibrator import save_samples
import os


DASHBOARD_BASE_PORT = 5800

# match to v4l2-ctl --list-ctrls
EXPOSURE_SLIDER_MAX = 100  # v4l2 units of 0.1ms
GAIN_SLIDER_MAX = 255

STREAM_MAX_FPS = 15


def dashboard_port(camera_id: int) -> int:
    return DASHBOARD_BASE_PORT + camera_id


MJPEG_PART_HEADER = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'


def mjpeg_part(jpeg: bytes) -> bytes:
    return jpeg + b'\r\n' + MJPEG_PART_HEADER


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
        
        pipeline_type = pipeline.get_pipeline_type()
        if pipeline_type == "raw":
            settings_panel = self.create_raw_capture_panel(cam_id)
            side_title = "Raw Capture"
        elif pipeline_type == "apriltag":
            settings_panel = self.create_apriltag_sliders(cam_id)
            side_title = "Detections"
        else:
            settings_panel = self.create_gamepiece_sliders(cam_id)
            side_title = "Detections"
        
        return html.Div([
                html.Div([
                    html.Div([
                        html.H4(side_title, style={
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
                                'max-height': '310px',
                                'overflow-y': 'auto',
                                'overflow-x': 'hidden',
                                'flex-shrink': '0',
                            }
                        ),
                        html.Br(),
                        settings_panel,
                        self.create_exposure_controls(cam_id),
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
                            html.Div(id=f'system-{cam_id}', style={
                                "position": "relative",
                                "width": "100%",
                                "max-width": "600px",
                                "height": "32px",
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

    def create_camera_switcher(self):
        camera_buttons = []
        for cam_id, pipeline in sorted(self.vision_manager.get_all_pipelines().items()):
            is_current = cam_id == self.camera_id
            label = f"{pipeline.get_camera_name()} ({cam_id})"
            if not pipeline.is_enabled():
                label += " · off"
            style = {
                "display": "inline-block",
                "margin": "0 8px 8px 0",
                "padding": "8px 14px",
                "border-radius": "16px",
                "font-size": "13px",
                "font-weight": "bold",
                "text-decoration": "none",
                "color": "#161616" if is_current else "#CCC9CA",
                "background-color": "#CDA646" if is_current else "rgba(255,255,255,0.08)",
                "border": "1px solid #CDA646" if is_current else "1px solid rgba(255,255,255,0.2)",
                "cursor": "pointer",
            }
            if is_current:
                camera_buttons.append(html.Span(label, style=style))
            else:
                camera_buttons.append(html.A(
                    label,
                    href="#",
                    className="camera-nav-link",
                    **{"data-port": str(dashboard_port(cam_id))},
                    style=style,
                ))

        return html.Div([
            html.Div("Cameras", style={
                "color": "#CCC9CA",
                "font-size": "12px",
                "font-weight": "bold",
                "margin-bottom": "8px",
                "letter-spacing": "0.04em",
                "text-transform": "uppercase",
            }),
            html.Div([
                html.A(
                    "All Cameras",
                    href="#",
                    className="camera-hub-link",
                    **{"data-port": str(DASHBOARD_BASE_PORT)},
                    style={
                        "display": "inline-block",
                        "margin": "0 8px 8px 0",
                        "padding": "8px 14px",
                        "border-radius": "16px",
                        "font-size": "13px",
                        "font-weight": "bold",
                        "text-decoration": "none",
                        "color": "#161616",
                        "background-color": "rgba(100, 200, 255, 1)",
                    },
                ),
                *camera_buttons,
            ]),
        ], style={
            "margin": "0 25px 15px 25px",
            "padding": "12px 0 4px 0",
            "border-bottom": "1px solid rgba(255,255,255,0.1)",
        })

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

            self.create_camera_switcher(),
            
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

    def _capture_button_style(self, bg_color):
        return {
            "margin": "8px 5px",
            "font-size": "14px",
            "color": "#161616",
            "background-color": bg_color,
            "border": "none",
            "padding": "8px 16px",
            "width": "200px",
            "height": "40px",
            "border-radius": "20px",
            "cursor": "pointer",
            "font-weight": "bold",
        }

    def create_raw_capture_panel(self, cam_id):
        return html.Div([
            html.H4("Capture", style={
                'textAlign': 'left',
                'color': '#CCC9CA',
                'font-size': '18px',
                'font-weight': 'medium',
                'padding': '15px 0px 0px 7px',
            }),
            html.P(
                "Live preview is the raw camera frame (no preprocess, no annotations). "
                "Snapshots save as PNG; recordings save as MJPEG AVI under captures/.",
                style={
                    'color': '#CCC9CA',
                    'font-size': '14px',
                    'padding': '10px 15px',
                    'margin': '0',
                },
            ),
            html.Div([
                html.Button(
                    "Snapshot",
                    id={'type': 'raw-snapshot', 'index': cam_id},
                    style=self._capture_button_style("rgba(100, 200, 255, 1)"),
                ),
                html.Button(
                    "Start Recording",
                    id={'type': 'raw-record-start', 'index': cam_id},
                    style=self._capture_button_style("rgba(100, 200, 100, 1)"),
                ),
                html.Button(
                    "Stop Recording",
                    id={'type': 'raw-record-stop', 'index': cam_id},
                    style=self._capture_button_style("rgba(255, 100, 100, 1)"),
                ),
            ], style={
                "display": "flex",
                "flex-direction": "column",
                "align-items": "center",
                "padding": "10px",
            }),
            html.Div(
                id=f'capture-status-{cam_id}',
                style={
                    'color': '#CCC9CA',
                    'font-size': '14px',
                    'padding': '15px 20px',
                    'margin': '15px 7px',
                    'border': '2px solid rgba(255, 255, 255, 0.3)',
                    'border-radius': '10px',
                },
            ),
        ])

    def create_exposure_controls(self, cam_id):
        pipeline = self.vision_manager.get_pipeline(cam_id)
        if not pipeline:
            return html.Div()
        settings = pipeline.cam.get_exposure_settings()
        auto = settings["auto_exposure"]

        resolution_controls = []
        if pipeline.cam.use_preprocessing:
            width, height = pipeline.cam.get_resolution()
            resolution_controls = [
                html.Label("Resolution", style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    'padding': '10px 20px 0 15px',
                }),
                dcc.RadioItems(
                    id={'type': 'resolution', 'index': cam_id},
                    options=[{'label': f' {w}x{h}', 'value': f'{w}x{h}'} for w, h in RESOLUTIONS],
                    value=f'{width}x{height}',
                    inline=True,
                    labelStyle={'margin-right': '20px'},
                    style={
                        "color": "#CCC9CA",
                        "font-size": "16px",
                        'padding': '10px 20px 0 15px',
                    },
                ),
                html.Div(self._resolution_status(cam_id, width, height), id={'type': 'resolution-status', 'index': cam_id}, style={
                    'color': '#CCC9CA',
                    'font-size': '13px',
                    'padding': '5px 20px 0 15px',
                }),
            ]

        return html.Div([
            html.H4("Camera", style={
                'textAlign': 'left',
                'color': '#CCC9CA',
                'font-size': '18px',
                'font-weight': 'medium',
                'padding': '15px 0px 0px 7px',
            }),

            *resolution_controls,

            dcc.Checklist(
                id={'type': 'auto-exposure', 'index': cam_id},
                options=[{'label': ' Auto Exposure', 'value': 'auto'}],
                value=['auto'] if auto else [],
                style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    'padding': '10px 20px 15px 15px',
                },
            ),

            html.Label("Exposure (x100 µs)", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'exposure', 'index': cam_id},
                min=1,
                max=EXPOSURE_SLIDER_MAX,
                step=1,
                value=min(settings["exposure"], EXPOSURE_SLIDER_MAX),
                disabled=auto,
                marks={1: '1', EXPOSURE_SLIDER_MAX: str(EXPOSURE_SLIDER_MAX)},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),

            html.Label("Gain", style={
                "color": "#CCC9CA",
                "font-size": "16px",
                'padding': '0 20px 0 15px',
            }),
            dcc.Slider(
                id={'type': 'gain', 'index': cam_id},
                min=0,
                max=GAIN_SLIDER_MAX,
                step=1,
                value=min(settings["gain"], GAIN_SLIDER_MAX),
                disabled=auto,
                marks={0: '0', GAIN_SLIDER_MAX: str(GAIN_SLIDER_MAX)},
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider"
            ),
            html.Br(),

            html.Div(settings["status"], id={'type': 'exposure-status', 'index': cam_id}, style={
                'color': '#CCC9CA',
                'font-size': '13px',
                'padding': '15px 20px 0 15px',
            }),
        ])

    def _resolution_status(self, cam_id, width, height):
        if has_opencv_calibration(cam_id, width, height):
            return f"Using {width}x{height} calibration"
        return f"No calibration at {width}x{height}: tags won't report distance until you calibrate on the Calibration tab"

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

                    dcc.Checklist(
                        id={'type': 'save-cal-samples', 'index': cam_id},
                        options=[{'label': ' Save sample images (to calibrate on another computer)', 'value': 'save'}],
                        value=['save'] if save_samples.valueInt() else [],
                        style={
                            "color": "#CCC9CA",
                            "font-size": "16px",
                            'padding': '0 20px 0 15px',
                        },
                    ),
                    
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
                    html.P("use a longer exposure with low gain so sensor noise doesn't inflate the score. Switch back to match settings after", style={
                        'color': '#CCC9CA',
                        'font-size': '14px',
                        'padding': '0 20px',
                        'margin': '0 7px',
                    }),
                    self.create_exposure_controls(cam_id),
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
            Output(f'system-{cam_id}', 'children'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_system_stats(n_intervals):
            stats = get_system_stats()
            if stats is None:
                return [html.Span("CPU stats need Linux", style={'color': '#888', 'padding': '5px 10px'})]
            stat_style = {
                "position": "absolute",
                "bottom": "0",
                "color": "rgba(255, 255, 255, 0.8)",
                "font-size": "16px",
                "font-style": "italic",
                "padding": "5px 10px",
            }
            per_core = " · ".join(f"cpu{i} {p:.0f}%" for i, p in enumerate(stats["cores"]))
            temp = stats["temp_c"]
            return [
                html.Span(f"CPU: {stats['cpu']:.0f}% (busiest core {max(stats['cores'], default=0):.0f}%)",
                          title=per_core, style={**stat_style, "left": "0"}),
                html.Span(f"CPU Temp: {temp:.1f} °C" if temp is not None else "CPU Temp: n/a",
                          style={**stat_style, "right": "0"}),
            ]

        @self.app.callback(
            Output(f'detections-{cam_id}', 'children'),
            [Input('update-interval', 'n_intervals')]
        )
        def update_detections(n_intervals, camera_id=cam_id):
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if not pipeline or not pipeline.is_enabled():
                return [html.Div("Camera disabled", style={'color': '#888'})]

            if pipeline.get_pipeline_type() == "raw":
                status = pipeline.get_capture_status()
                recording_label = "Recording" if status["recording"] else "Idle"
                color = "#00ff00" if status["recording"] else "#CCC9CA"
                return [html.Div([
                    html.Div(f"Mode: raw", style={'color': '#CCC9CA', 'margin': '5px 0'}),
                    html.Div(f"Status: {recording_label}", style={'color': color, 'margin': '5px 0'}),
                    html.Div(
                        status["message"] or "Use Snapshot / Start Recording",
                        style={'color': '#CCC9CA', 'margin': '5px 0', 'font-size': '13px'},
                    ),
                ], style={
                    'border': '2px solid rgba(255, 255, 255, 0.5)',
                    'border-radius': '10px',
                    'padding': '10px',
                    'margin': '0 0px 20px 20px',
                })]

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
                    }))
            return detection_items
    
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
            [Output({'type': 'exposure', 'index': MATCH}, 'disabled'),
             Output({'type': 'gain', 'index': MATCH}, 'disabled'),
             Output({'type': 'exposure-status', 'index': MATCH}, 'children')],
            [Input({'type': 'auto-exposure', 'index': MATCH}, 'value'),
             Input({'type': 'exposure', 'index': MATCH}, 'value'),
             Input({'type': 'gain', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_exposure(auto_value, exposure, gain):
            auto = 'auto' in (auto_value or [])
            pipeline = self.vision_manager.get_pipeline(self.camera_id)
            if not pipeline or exposure is None or gain is None:
                return auto, auto, no_update
            pipeline.cam.set_exposure_settings(auto, int(exposure), int(gain))
            return auto, auto, pipeline.cam.get_exposure_settings()["status"]

        @self.app.callback(
            Output({'type': 'resolution-status', 'index': MATCH}, 'children'),
            [Input({'type': 'resolution', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_resolution(value):
            pipeline = self.vision_manager.get_pipeline(self.camera_id)
            if not pipeline or not value:
                return no_update
            width, height = map(int, value.split("x"))
            if (width, height) != pipeline.cam.get_resolution():
                pipeline.cam.set_resolution(width, height)
                if self._calibrator is not None:
                    self._calibrator.reset()
            return self._resolution_status(self.camera_id, width, height)

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

        @self.app.callback(
            Output({'type': 'raw-snapshot', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'raw-snapshot', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def raw_snapshot(n_clicks):
            if n_clicks:
                pipeline = self.vision_manager.get_pipeline(self.camera_id)
                if pipeline:
                    pipeline.save_snapshot()
            return n_clicks

        @self.app.callback(
            Output({'type': 'raw-record-start', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'raw-record-start', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def raw_record_start(n_clicks):
            if n_clicks:
                pipeline = self.vision_manager.get_pipeline(self.camera_id)
                if pipeline:
                    pipeline.start_recording()
            return n_clicks

        @self.app.callback(
            Output({'type': 'raw-record-stop', 'index': MATCH}, 'n_clicks'),
            [Input({'type': 'raw-record-stop', 'index': MATCH}, 'n_clicks')],
            prevent_initial_call=True
        )
        def raw_record_stop(n_clicks):
            if n_clicks:
                pipeline = self.vision_manager.get_pipeline(self.camera_id)
                if pipeline:
                    pipeline.stop_recording()
            return n_clicks

        @self.app.callback(
            Output(f'capture-status-{self.camera_id}', 'children'),
            [Input('update-interval', 'n_intervals')],
        )
        def update_capture_status(n_intervals):
            pipeline = self.vision_manager.get_pipeline(self.camera_id)
            if not pipeline or pipeline.get_pipeline_type() != "raw":
                return no_update
            status = pipeline.get_capture_status()
            recording = "Recording" if status["recording"] else "Idle"
            path = status["path"] or "—"
            message = status["message"] or "Ready"
            return html.Div([
                html.Div(f"Recorder: {recording}", style={'margin': '5px 0'}),
                html.Div(f"File: {path}", style={'margin': '5px 0'}),
                html.Div(f"Last: {message}", style={'margin': '5px 0'}),
            ])
        
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
                html.Div(f"Images: {status['sample_dir']}", style={'margin': '5px 0'}) if status['sample_dir'] else html.Div(),
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
            Output({'type': 'save-cal-samples', 'index': MATCH}, 'value'),
            [Input({'type': 'save-cal-samples', 'index': MATCH}, 'value')],
            prevent_initial_call=True
        )
        def update_save_samples(value):
            save_samples.setInt(1 if 'save' in (value or []) else 0)
            return value

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
        last_source = None
        next_send = 0.0

        yield MJPEG_PART_HEADER
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

                yield mjpeg_part(blank_frame_cache)
                time.sleep(0.1)
                continue

            time.sleep(max(0.0, next_send - time.monotonic()))
            frame = pipeline.get_frame()
            if frame is None or frame is last_source:
                time.sleep(0.01)
                continue
            last_source = frame
            next_send = time.monotonic() + 1.0 / STREAM_MAX_FPS

            quality = int(DashboardServer.framecomp_slider.valueFloat() * 100)
            if pipeline.get_pipeline_type() == "raw":
                quality = max(quality, 95)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            yield mjpeg_part(buffer.tobytes())

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
        port = dashboard_port(self.camera_id)
        print(f"\n{'='*50}")
        print(f"MonkeySee Dashboard Starting for Camera {self.camera_id}")
        print(f"Port: {port}")
        print(f"Access at: http://0.0.0.0:{port}")
        print(f"{'='*50}\n")
        self.app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

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
            <script>
                document.addEventListener('click', function(e) {
                    var el = e.target.closest('.camera-nav-link, .camera-hub-link');
                    if (!el) return;
                    var port = el.getAttribute('data-port');
                    if (!port) return;
                    e.preventDefault();
                    window.location.href = 'http://' + window.location.hostname + ':' + port + '/';
                });
            </script>
        </head>
        <body>
            {%app_entry%}
            {%config%}
            {%scripts%}
            {%renderer%}
        </body>
        </html>
        """


class CameraIndexServer:
    """Hub on port 5800 listing every camera stream with capture controls."""

    def __init__(self, vision_manager: VisionManager):
        self.vision_manager = vision_manager
        self.server = Flask("MonkeySeeCameraIndex")
        self.setup_routes()
        Thread(target=self.start_server, daemon=True).start()

    def start_server(self):
        print(f"\n{'='*50}")
        print("MonkeySee Camera Hub Starting")
        print(f"Port: {DASHBOARD_BASE_PORT}")
        print(f"Access at: http://0.0.0.0:{DASHBOARD_BASE_PORT}")
        print(f"{'='*50}\n")
        self.server.run(host="0.0.0.0", port=DASHBOARD_BASE_PORT, debug=False, use_reloader=False)

    def setup_routes(self):
        @self.server.route("/")
        def index():
            return self._index_html()

        @self.server.route("/api/cameras")
        def api_cameras():
            cameras = []
            for cam_id, pipeline in sorted(self.vision_manager.get_all_pipelines().items()):
                status = pipeline.get_capture_status()
                cameras.append({
                    "id": cam_id,
                    "name": pipeline.get_camera_name(),
                    "pipeline": pipeline.get_pipeline_type(),
                    "enabled": pipeline.is_enabled(),
                    "dashboard_port": dashboard_port(cam_id),
                    "recording": status["recording"],
                    "message": status["message"],
                    "path": status["path"],
                })
            return jsonify({"cameras": cameras})

        @self.server.route("/video_feed/<int:camera_id>")
        def video_feed(camera_id: int):
            return Response(
                self._generate_frames(camera_id),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.server.route("/api/<int:camera_id>/snapshot", methods=["POST"])
        def api_snapshot(camera_id: int):
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if not pipeline:
                return jsonify({"ok": False, "message": "Camera not found"}), 404
            path = pipeline.save_snapshot()
            status = pipeline.get_capture_status()
            return jsonify({"ok": path is not None, "path": path, **status})

        @self.server.route("/api/<int:camera_id>/record/start", methods=["POST"])
        def api_record_start(camera_id: int):
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if not pipeline:
                return jsonify({"ok": False, "message": "Camera not found"}), 404
            message = pipeline.start_recording()
            status = pipeline.get_capture_status()
            return jsonify({"ok": True, "message": message, **status})

        @self.server.route("/api/<int:camera_id>/record/stop", methods=["POST"])
        def api_record_stop(camera_id: int):
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if not pipeline:
                return jsonify({"ok": False, "message": "Camera not found"}), 404
            message = pipeline.stop_recording()
            status = pipeline.get_capture_status()
            return jsonify({"ok": True, "message": message, **status})

    def _generate_frames(self, camera_id: int):
        blank_frame_cache = None
        last_source = None
        next_send = 0.0
        yield MJPEG_PART_HEADER
        while True:
            pipeline = self.vision_manager.get_pipeline(camera_id)
            if not pipeline or not pipeline.is_enabled():
                if blank_frame_cache is None:
                    blank = 255 * cv2.ones((480, 640, 3), dtype="uint8")
                    cv2.putText(
                        blank,
                        "Camera Unavailable",
                        (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (100, 100, 100),
                        2,
                    )
                    ret, buffer = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    blank_frame_cache = buffer.tobytes()
                yield mjpeg_part(blank_frame_cache)
                time.sleep(0.1)
                continue

            time.sleep(max(0.0, next_send - time.monotonic()))
            frame = pipeline.get_frame()
            if frame is None or frame is last_source:
                time.sleep(0.01)
                continue
            last_source = frame
            next_send = time.monotonic() + 1.0 / STREAM_MAX_FPS

            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield mjpeg_part(buffer.tobytes())

    def _index_html(self):
        cards = []
        for cam_id, pipeline in sorted(self.vision_manager.get_all_pipelines().items()):
            enabled = pipeline.is_enabled()
            status = "Live" if enabled else "Disabled"
            cards.append(
                f"""
                <article class="card" data-cam="{cam_id}">
                  <div class="card-head">
                    <div>
                      <h2>{pipeline.get_camera_name()}</h2>
                      <p class="meta">Camera {cam_id} · {pipeline.get_pipeline_type()} · {status}</p>
                    </div>
                    <a class="dash-link" href="http://HOST:{dashboard_port(cam_id)}/">Open dashboard</a>
                  </div>
                  <img class="stream" src="/video_feed/{cam_id}" alt="Camera {cam_id} stream" />
                  <div class="actions">
                    <button type="button" data-action="snapshot" data-cam="{cam_id}">Snapshot</button>
                    <button type="button" data-action="record-start" data-cam="{cam_id}">Start Recording</button>
                    <button type="button" data-action="record-stop" data-cam="{cam_id}">Stop Recording</button>
                  </div>
                  <p class="status" id="status-{cam_id}">Ready</p>
                </article>
                """
            )

        cards_html = "\n".join(cards) if cards else "<p class='empty'>No cameras found in config.</p>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MonkeyVision Cameras</title>
  <style>
    :root {{
      --bg: #161616;
      --text: #CCC9CA;
      --accent: #CDA646;
      --panel: #1f1f1f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 32px;
      color: var(--text);
    }}
    .subtitle {{
      margin: 0 0 24px;
      color: var(--text);
      opacity: 0.85;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }}
    .card h2 {{
      margin: 0;
      font-size: 18px;
      color: var(--accent);
    }}
    .meta {{
      margin: 4px 0 0;
      font-size: 13px;
      opacity: 0.8;
    }}
    .dash-link {{
      color: #161616;
      background: var(--accent);
      text-decoration: none;
      font-weight: 700;
      font-size: 12px;
      padding: 8px 12px;
      border-radius: 16px;
      white-space: nowrap;
    }}
    .stream {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #000;
      border: 2px solid var(--accent);
      border-radius: 9px;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    button {{
      border: none;
      border-radius: 18px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      color: #161616;
      background: rgba(100, 200, 255, 1);
    }}
    button[data-action="record-start"] {{ background: rgba(100, 200, 100, 1); }}
    button[data-action="record-stop"] {{ background: rgba(255, 100, 100, 1); }}
    .status {{
      margin: 0;
      min-height: 1.2em;
      font-size: 13px;
      opacity: 0.9;
    }}
    .empty {{ opacity: 0.7; }}
  </style>
</head>
<body>
  <h1>MonkeyVision</h1>
  <p class="subtitle">Pick a camera stream to preview, snapshot, or record. Snapshots/recordings save under captures/.</p>
  <div class="grid">
    {cards_html}
  </div>
  <script>
    (function() {{
      const host = window.location.hostname;
      document.querySelectorAll('.dash-link').forEach(function(el) {{
        el.href = el.href.replace('HOST', host);
      }});

      async function refreshStatuses() {{
        try {{
          const res = await fetch('/api/cameras');
          const data = await res.json();
          (data.cameras || []).forEach(function(cam) {{
            const el = document.getElementById('status-' + cam.id);
            if (!el) return;
            const state = cam.recording ? 'Recording' : 'Idle';
            const msg = cam.message || 'Ready';
            el.textContent = state + ' · ' + msg;
          }});
        }} catch (e) {{}}
      }}

      document.querySelectorAll('button[data-action]').forEach(function(btn) {{
        btn.addEventListener('click', async function() {{
          const cam = btn.getAttribute('data-cam');
          const action = btn.getAttribute('data-action');
          let url = '/api/' + cam + '/snapshot';
          if (action === 'record-start') url = '/api/' + cam + '/record/start';
          if (action === 'record-stop') url = '/api/' + cam + '/record/stop';
          try {{
            const res = await fetch(url, {{ method: 'POST' }});
            const data = await res.json();
            const el = document.getElementById('status-' + cam);
            if (el) el.textContent = data.message || (data.ok ? 'OK' : 'Failed');
          }} catch (e) {{
            const el = document.getElementById('status-' + cam);
            if (el) el.textContent = 'Request failed';
          }}
        }});
      }});

      refreshStatuses();
      setInterval(refreshStatuses, 1000);
    }})();
  </script>
</body>
</html>
"""

