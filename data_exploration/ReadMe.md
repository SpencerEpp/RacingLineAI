# Assetto Corsa Track Extraction Pipeline
This pipeline converts Assetto Corsa .kn5 track files and ai-spline files into structured, ML-friendly data. It extracts:<br/>
- The **racing line** from ideal_lane.ai.<br/>
- The **road mesh geometry** from .kn5 track files.<br/>
- The **left and right boundaries** of the track from meshes in the .kn5 file, estimated perpendicularly from the AI center line fast_lane.ai.<br/>

#### Install Requirements
Please do the following before running these files.<br/> 
1. python -m venv venv<br/> 
2. source venv/bin/activate<br/> 
3. pip install -r requirements.txt<br/> 

#### Pipeline Overview
1. Scans each track folder in the Assetto Corsa directory.<br/> 
2. Finds the correct .kn5 file matching the track name.<br/> 
3. Identifies layout pairs by locating matching fast_lane.ai and ideal_line.ai files in either the root or layout subfolders.<br/> 
4. Parses the fast_lane.ai as the centerline and extracts road mesh vertices from the .kn5.<br/> 
5. Estimates left and right track edges by projecting perpendicular to the centerline and snapping to the nearest mesh vertices.<br/> 
6. Parses the ideal_line.ai as the driving target.<br/> 
7. Saves both ideal_line.csv and track_edges.csv into layout-specific folders for each track.<br/> 

#### Required General Structure
/RacingLineAI/<br/> 
├── assetto_corsa_tracks/                 # original AC tracks with .kn5, fast_lane.ai, and ideal_lane.ai.<br/> 
│   └── track_name/<br/> 
│       ├── track_name.kn5<br/> 
│       ├── subfolder/ai/fast_lane.ai     # Note ai folders can be found in several sub folders or track root folder.<br/>
│       └── subfolder/data/ideal_lane.ai<br/>    
└── extracted_track_data/                 # final output goes here.<br/>

#### Output Structure
/RacingLineAI/data/extracted_track_data/<br/>
└── track_name/layout#/<br/>
    ├── ideal_line.csv       # 23 features that make up the ai driving line, Note multiple of these may be created if a track has several layouts.<br/>
    │                        # └── If several are created they are in the format <"layout">_<"layout_name">_<"racing_line">.csv<br/>
    └── track_edges.csv      # timestamp, left-x, left-y, left-z, right-x, right-y, right-z (estimated right edge from road mesh)<br/>

#### Notes
- Edge points are projected perpendicularly from the racing line and snapped to the nearest vertex on the road mesh only.<br/>
- Only mesh groups/materials containing keywords like road, asphalt, track or curb are used to build the KDTree for edge detection.<br/>
