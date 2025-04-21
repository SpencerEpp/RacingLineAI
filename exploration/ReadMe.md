# Assetto Corsa Track Extraction Pipeline
This pipeline converts Assetto Corsa .kn5 track files and ai-spline files into structured, ML-friendly data. It extracts:<br/>
- The **racing line** from ideal_lane.ai.<br/>
- The **road mesh geometry** from .kn5 track files.<br/>
- The **left and right boundaries** (x,z) of the track from map.png and fills in relatively accurate y from meshes in the .kn5 file, estimated perpendicularly from the AI center line fast_lane.ai.<br/>

#### Install Requirements
Please do the following before running these files.<br/> 
1. python -m venv venv<br/> 
2. source venv/bin/activate<br/> 
3. pip install -r requirements.txt<br/> 

#### Pipeline Overview
1. Scans each track folder in the Assetto Corsa directory.<br/> 
2. Finds the correct .png and .kn5 file matching the track name.<br/> 
3. Identifies layout pairs by locating matching fast_lane.ai and ideal_line.ai files in either the root or layout subfolders.<br/> 
4. Parses the fast_lane.ai as the centerline and extracts road mesh vertices from the .kn5.<br/> 
5. Estimates left and right track edges (x,z) using edge detection and y by projecting perpendicular to the centerline and snapping to the nearest mesh vertices.<br/> 
6. Parses the ideal_line.ai as the driving target.<br/> 
7. Saves all track data for training into csv's following the convention TrackName_LayoutName_Processed_Data.csv.<br/> 

#### Required General Structure
/RacingLineAI/<br/> 
├── assetto_corsa_tracks/                 # original AC tracks with .png, .kn5, fast_lane.ai, and ideal_lane.ai.<br/> 
│   └── track_name/<br/> 
│       ├── track_name.kn5<br/> 
│       ├── map.png                       # Note if multiple layouts map.png will be found in the subfolder for the layout.<br/>
│       ├── subfolder/map.png<br/> 
│       ├── subfolder/ai/fast_lane.ai     # Note ai folders can be found in several sub folders or track root folder.<br/>
│       └── subfolder/data/ideal_lane.ai<br/>    
└── extracted_track_data/                 # final output goes here.<br/>

#### Output Structure
/RacingLineAI/data/extracted_track_data/<br/>
└── TrackName_LayoutName_Processed_Data.csv<br/>

The csv contains the 6 input features and 24 target features all in one csv, in the following order:<br/>
left_x,left_y,left_z,right_x,right_y,right_z,x,y,z,length,id,speed,gas,brake,obsolete_lat_g,radius,side_left,side_right,camber,direction,normal_x,normal_y,normal_z,extra_length,forward_x,forward_y,forward_z,tag,grade<br/>

#### Notes
- Edge x,z points are constructed using edge detection on the map.png file.<br/>
- Edge y points are projected perpendicularly from the racing line and snapped to the nearest vertex on the road mesh only.<br/>
- Only mesh groups/materials containing keywords like road, asphalt, track or curb are used to build the KDTree for edge detection.<br/>
- Included tracks up to page 5 from https://www.assettohub.com/tracks/?_pagination=5 on April 19th