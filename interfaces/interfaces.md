# Interface definitions for FIM Evaluation Jobs Arguments, Inputs, and Outputs

This interface directory contains a set of yaml files that specifies interfaces for the jobs that make up the HAND FIM evaluation pipeline. The pipeline has been designed as a series of chained jobs that will be run by a batch processing solution. The primary focus of this repo is on describing inputs and outputs of each job and each jobs arguments/parameters. 

Below we give a human readable description of the contents of each yaml file. The descriptions also expand on the anticipated behavior of the jobs for common input and argument combinations. 

By convention when outputs are listed for a job it is assumed that these outputs will always be written to a filepath(s) that is specified when a job is called. This is to make it easier to integrate the jobs with a batch orchestrator. std-out is reserved for job logging rather than output. 

## HAND Inundator (`hand_inundator`)

**Implementation status:  partially implemented. Depth FIM production will be added in PI-6**

### Example docker run command

From inside the inundate-dev container would run:

```
python inundate.py --region_tag 1234 --catchment_data_path /path/to/catchment/json/ --forecast_path /path/to/forecast/ --output_path /path/to/output/ --fim_type extent --geo_mem_cache 512
```


### Description  
- Generates flood extent/depth maps from a HAND REM. This job inundates a *single* hand catchment. It can be configured to return either a depth FIM or an extent FIM.

### Arguments  
- **geo_mem_cache**
  - The GDAL cache size in megabytes to use when processing gridded data. Raster processing is the most memory intensive portion of this job so this argument effectively limits the memory used by the job.
- **output_type**
  - Extent (binary) vs Depth (float values)  

### Inputs 
- **catchment**:
  - This input is a JSON file that contains a rating curve every HydroID in a HAND catchment along with metadata necessary to process the HydroID. It has the following structure:
  ```json
  {
      "<catchment_id>": {
        "hydrotable_entries": {
          "<HydroID>": {
            "stage": ["array_of_stage_values"],
            "discharge_cms": ["array_of_discharge_values"],
            "nwm_feature_id": "<integer>",
            "lake_id": "<integer>"
          }
          // More HydroID entries...
        },
        "raster_pair": {
          "rem_raster_path": "<path_value>",
          "catchment_raster_path": "<path_value>"
        }
      }
  }
  ```
  - The **raster_pair** lists paths to the two rasters that are used to generate the HAND extent.
    - **rem_raster_path**
    - This is a path to a HAND relative elevation tiff for this catchment. This would typically be an s3 path but could be a local filepath as well. 
    - **catchment_raster_path**
    - This is a path to a tiff that helps map every location in the catchment to a rating curve associated with that location. Every pixel is assigned an integer value that reflects the HydroID of the sub-catchment it is in. This value can then be used to look up an associated rating curve in the hydrotable_entries object inside the catchment json. This rating curve is used to interpolate a stage value for a given NWM reach discharge. If the stage value is larger than the HAND value at that pixel then the pixel is marked flooded.
- **flow_scenario**
  - A csv file listing NWM feature_id values and their respective discharges. A stage is obtained for these discharges for each HydroID catchment by using the rating associated with that HydroID.

### Outputs 
- **Inundation Raster**
  - This is a depth or extent raster generated from the HAND data. The format of this raster is specified in `hand_inundator.yml'
---

## Mosaic Maker (`fim_mosaicker`) 

**Implementation status:  partially implemented. Vector mosaicking and mosaicking for depth rasters will be added in PI-6**

### Example command

From inside the mosaic-dev container would run:

```
python mosaic.py --region_tag 1234 --raster_paths /paths/to/rasters/ --output_path /path/to/output/ --clip_geometry /path/to/clipvectors --fim_type extent --resolution 3 --geo_mem_cache 512
```

### Description  
This job mosaics flood extents and benchmark raster data from either HAND or benchmark sources using a pixel-wise NAN-MAX selection policy. That is, for all the images being mosaicked if there are overlapping raster pixels then the maximum value of the overlapping rasters at that pixel location is selected. No-Data values are not considered when selecting the maximum (they are treated as Nan) unless all the pixels are No-Data. Rasters can be either depth or extent rasters and the mosaicking policy for overlapping rasters will remain the same. Common input combinations and the behavior of the jobs in those cases are described below.

If vector data is being mosaicked then the behavior will depend on what type of geometry is described by the vector data. The allowed geometries are polygon, multipolygon, point, or multipoint.  

In the case of a polygon or multipolygon geometries they will be treated as only describing an extent. Any point geometries mosaicked with them will be cast to extents. Rasters mosaicked with these geometries will be cast to polygonal extents as well and the final output will a mosaicked polygon, multipolygon, or polygon with adjacent points.

In the case of point or multipoint geometries they can contain either extents or depths in their attributes and will pass through the type of information they are tagged with unless they are being mosaicked with polygon geometries. When a raster is being mosaicked with point geometries, locations where the raster coincide with the point values will be converted to a point geometry by averaging extent or depth pixel values within a buffer of the point location. This value will then be mosaicked with the value described by the overlapping point geometries depth or extent attribute. The raster values that don't coincide with points in the point geometry will be discarded. Additional attributes from the original point geometries will be passed through and returned with the mosaicked geometries.

### Arguments
- **target_resolution**
  - Resolution of the final mosaicked FIM when a raster will be produced. Required for raster outputs  
- **fim_type**
  - This informs the job whether it is mosaicking FIMs with extents or depths.
- **geo_mem_cache**
  - The GDAL cache size in megabytes to use when processing gridded data. Raster processing is the most memory intensive portion of this job so this argument effectively limits the memory used by the job.

### Inputs
- **Files**: 
  - Array of paths to TIFF/GeoJSON/GeoPackage rasters and/or vector files. If a vector is listed in the array then the output will be a vector. 
- **Clipping Geometry**
  - Optional GeoJSON boundary to clip the mosaicked output to. This input will always be given in the HAND FIM evaluation pipeline and will describe the ROI being evaluated.

### Outputs 
- **Raster**
  - In the case of raster output, the output will be a single mosaicked raster.
- **Vector** 
  - In the case of vector output, the output will be a single mosaicked vector tiff file.

---

## Agreement Maker (`agreement_maker`) 

**Implementation status:  Will be implemented in NGWPC PI-6**

### Example command

From inside the agreement-dev container would run:

```
python agreement.py --region_tag 1234 --benchmark_path /path/to/raster/or/multipoint --candidate_path /path/to/raster/or/multipoint --agreement_path /path/to/agreement/ --clip_geometry /path/to/clipvectors --fim_type extent --resolution 3 --geo_mem_cache 512
```

### Description  
Creates an agreement map showing where a pair of input data (raster or vector) spatially concur. The job is designed to work with any combination of raster or vector input pairs. The job also works with depth or extent data with the assumption that a given pair will be either both depths or extents. Produces either a continuous  agreement map when the inputs are depths or a categorical agreement map for extents. The output is raster or vector data in EPSG:5070.

Similarly to the mosaicking job geometry inputs and outputs, polygon geometries can only describe extents.

### Arguments  
- **Resolution**
  - Mandatory x/y pixel resolution that is used when outputting rasters  

### Inputs
- **Dataset1/Dataset2**:  
  - Raster or Vector (as geopackage). If a vector must be either a point, multipoint, multipolygon, or polygon geometry.  

- **Mask dictionary**
  - This is an optional json object that is composed of sub-objects that include paths to geopackage of masks to exclude or include in the final produced agreement. The input format is identical to the previous format that was previously used to mask areas over which to evaluate FIM model skill. Each mask geometry can also be buffered by setting a buffer flag to an integer value (with units of meters) in the sub-dictionaries "buffer" key.

  ```json
  {
    "levees": {
      "path": "path/to/levee/file",
      "buffer": null,
      "operation": "exclude"
    },
    "waterbodies": {
      "path": "path/to/waterbody/file",
      "buffer": null,
      "operation": "exclude"
    }
  }
  ```
  
### Outputs 
Output is either a single raster or a geopackage of vector information.

- **Raster**
  - See `agreement_maker.yml` for a description of the output raster format.
- **Vector**: 
  - See `agreement_maker.yml` for a description of output vector format. The returned geopackage could have additional attributes that are passed through from the input vector data to the output data. 

---

## Metrics Calculator (`metrics_calculator`) 

**Implementation status:  Will be implemented in NGWPC PI-6**

### Example command

From inside the metrics-dev container would run:

```
python metrics.py --region_tag 1234 --agreement_path /path/to/agreement/ --metrics_path /path/to/metrics/json --geo_mem_cache 512
```

### Description  
This job is designed to take an agreement map and calculate summary metrics of the agreement of two FIMs over a given ROI.

### Input  
- **Agreement map**
  - Accepts raster TIFF or geopackage containing point or polygon geometries. The type of agreement map will be surmised by the number and type of data values in the raster or the geometry's attributes. 

### Output  
- **Metrics json**
  - The output will be a json file containing the metrics the user requested. `metrics_calculator.yml` lists a small subset of possible metrics.
---

