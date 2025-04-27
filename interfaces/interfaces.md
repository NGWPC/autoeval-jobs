# Interface definitions for FIM Evaluation Jobs Arguments, Inputs, and Outputs

This interface directory contains a set of yaml files that specifies interfaces for the jobs that make up the HAND FIM evaluation pipeline. The pipeline has been designed as a series of chained jobs that will be run by a batch processing solution. The primary focus of this repo is on describing inputs and outputs of each job and each jobs arguments/parameters. 

Below we give a human readable description of the contents of each yaml file. The descriptions also expand on the anticipated behavior of the jobs for common input and argument combinations. 

By convention when outputs are listed for a job it is assumed that these outputs will always be written to a filepath(s) that is specified when a job is called. This is to make it easier to integrate the jobs with a batch orchestrator. std-out is reserved for job logging rather than output. 

## HAND Inundator (`hand_inundator`)

**Implementation status: Inundated extents implemented. Depth FIM production will be added in FY26**

### Example docker run command

From inside the inundate-dev container would run:

```
python inundate.py --catchment_data_path /path/to/catchment/json/ --forecast_path /path/to/forecast/ --fim_output_path /path/to/output/ --fim_type extent
```

This example lists all possible arguments. See yaml files for optional vs required arguments.

### Description  
- Generates flood extent/depth maps from a HAND REM. This job inundates a *single* hand catchment. It can be configured to return either a depth FIM or an extent FIM.

### Arguments  
- **fim_type**
  - Extent (binary) vs Depth (float values)  

### Inputs 
- **catchment_data_path**:
  - This input is path to a JSON file that contains a rating curve every HydroID in a HAND catchment along with metadata necessary to process the HydroID. The json file should have the following structure:
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
- **forecast_path**
  - A path to a csv file listing NWM feature_id values and their respective discharges. A stage is obtained for these discharges for each HydroID catchment by using the rating associated with that HydroID.

### Outputs 
- **fim_output_path**
  - This is a depth or extent raster generated from the HAND data depending on the value of the fim_type argument. The format of this raster is specified in `hand_inundator.yml'

---

## Mosaic Maker (`fim_mosaicker`) 

**Implementation status: Partially implemented. The depth FIM functionality will be implemented in FY26.**

### Example command

From inside the mosaic-dev container would run:

```
python mosaic.py --raster_paths /paths/to/rasters/ --hwm_paths /path/to/multipoint/geometries --mosaic_output_path /path/to/output/ --clip_geometry /path/to/clipvectors --fim_type extent
```

This example lists all possible arguments. See yaml files for optional vs required arguments.

### Description  
This job mosaics flood extents and benchmark raster data from either HAND or benchmark sources using a pixel-wise NAN-MAX selection policy. That is, for all the images being mosaicked if there are overlapping raster pixels then the maximum value of the overlapping rasters at that pixel location is selected. No-Data values are not considered when selecting the maximum (they are treated as Nan) unless all the pixels are No-Data. Rasters can be either depth or extent rasters and the mosaicking policy for overlapping rasters will remain the same. The resolution of the produced raster will be determined by the lowest resolution raster in the input data.

### Arguments
- **fim_type**
  - This informs the job whether it is mosaicking FIMs with extents or depths.

### Inputs
- **raster_paths** 
  - An array of paths to rasters in tiff format. The array should be a string formatted as a json list. This input can also be a file with the stringified json list inside it. 
- **clip_geometry_path**
  - Optional path to a GeoJSON or gpkg file with a boundary to clip the mosaicked output to. This input will always be given in the HAND FIM evaluation pipeline and will describe the ROI being evaluated.

### Outputs 
- **mosaic_output_path**
  - **Raster**
    - In the case of raster output, the output will be a path pointing to a single mosaicked raster.
  - **Vector** 
    - In the case of vector output, the output will be a path pointing to a single mosaicked vector gpkg.

---

## Agreement Maker (`agreement_maker`) 

**Implementation status:  Will be implemented in NGWPC PI-6. The depth FIM functionality will be implemented in FY26.**

### Example command

From inside the agreement-dev container would run:

```
python agreement.py --benchmark_path /path/to/raster/ --candidate_path /path/to/raster/ --agreement_path /path/to/agreement/ --clip_geoms /path/to/clipdictionary --fim_type extent 
```

This example lists all possible arguments. See yaml files for optional vs required arguments.

**Note on implementation memory usage:** The inundate and mosaicker jobs limit the memory used for raster processing by setting the GDAL_CACHEMAX environment variable. If the rioxarray based GVAL is used for the metrics_calculator job then a different argument or arguments will be needed to constrain the memory usage of the raster handling involved in the metrics calculation. If GVAL can't be made to limit its memory usage we will need to pursue a different approach.

### Description  
Creates an agreement map showing where a pair of input rasters spatially concur. The job works with depth or extent data with the assumption that a given pair will be either both depths or extents. Produces either a continuous agreement map when the inputs are depths or a categorical agreement map for extents. The resolution of the produced raster will be determined by the lowest resolution raster in the input data.


### Arguments  

- **fim_type**
  - Specifies whether agreement is based on spatial 'extent' (agreement between binary categorical rasters) or between rasters with depth values. Influences output raster format.
 
### Inputs
- **benchmark_path**:  
  - path to depth or extent raster benchmark data.  

- **candidate_path**:  
  - path to depth or extent raster benchmark data.   

- **clip_geoms**
  - This is an optional path to json file that that includes paths to geopackage of masks to exclude or include in the final produced agreement. The input format is identical to the previous format that was previously used to mask areas over which to evaluate FIM model skill. Each mask geometry can also be buffered by setting a buffer flag to an integer value (with units of meters) in the sub-dictionaries "buffer" key.

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
Output is a single raster 
- **agreement_path**
    - See `agreement_maker.yml` for a description of the output raster format for continuous or categorical agreement rasters.

---

## HWM Agreement Maker (`hwm_agreement`) 

**Implementation status:  Will be implemented in NGWPC PI-6**

### Example command

From inside the agreement-dev container would run:

```
python agreement.py --benchmark_path /path/to/multipoint --candidate_path /path/to/raster --agreement_path /path/to/agreement/ --clip_geoms /path/to/clipdictionary --fim_type extent 
```

This example lists all possible arguments. See yaml files for optional vs required arguments.

### Description  
Creates an agreement multipoint geometry showing where a FIM raster and a set of HWM points associated with an event spatially concur. The job works with depth or extent rasters with the assumption that a given HWM survey will the attributes required to produce an agreement map. The agreement geometry will be the same HWM point geometry with attributes indicating agreement between the HWM points and the raster being compared.

### Arguments  

- **fim_type**
  - Specifies whether agreement is based on spatial 'extent' overlap (binary) or potentially 'depth' values (requires specific logic in the script). Influences output raster format.
 
### Inputs
- **benchmark_path**:  
  - path to vector (as geopackage) benchmark data. ~must be either a point or multipoint geometry.  

- **candidate_path**:  
  - path to vector (as geopackage) benchmark data. If a vector must be either a point or multipoint geometry.  

- **clip_geoms**
  - This is an optional path to json file that that includes paths to geopackage of masks to exclude or include in the final produced agreement. The input format is identical to the previous format that was previously used to mask areas over which to evaluate FIM model skill. Each mask geometry can also be buffered by setting a buffer flag to an integer value (with units of meters) in the sub-dictionaries "buffer" key.

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
Output is a geopackage of vector information.
- **agreement_path**
    - See `agreement_maker.yml` for a description of output vector format. The returned geopackage could have additional attributes that are passed through from the input vector data to the output data. 

---

## Metrics Calculator (`metrics_calculator`) 

**Implementation status:  Will be implemented in NGWPC PI-6**


### Example command

From inside the metrics-dev container would run command below:

```
python metrics.py --agreement_path /path/to/agreement/ --metrics_path /path/to/metrics/json
```

This example lists all possible arguments. See yaml files for optional vs required arguments.

**Note on implementation memory usage:** The inundate and mosaicker jobs limit the memory used for raster processing by setting the GDAL_CACHEMAX environment variable. If the rioxarray based GVAL is used for the metrics_calculator job then a different argument or arguments will be needed to constrain the memory usage of the raster handling involved in the metrics calculation. If GVAL can't be made to limit its memory usage we will need to pursue a different approach.

### Description  
This job is designed to take an agreement map raster and calculate summary metrics of the agreement of two FIMs over a given ROI.

### Arguments  


### Input  
- **agreement_path**
  - Path to an agreement raster over which the metrics will be calculated.

### Output  
- **metrics_path**
  - The output will be a json file containing the metrics the user requested. `metrics_calculator.yml` lists a small subset of possible metrics.

---

##  HWM Metrics Calculator (`hwm_metrics`) 

### Example command

From inside the metrics-dev container would run command below:

```
python metrics.py --agreement_path /path/to/agreement/ --metrics_path /path/to/metrics/json
```

This example lists all possible arguments. See yaml files for optional vs required arguments.

### Description  
This job is designed to take an agreement map raster and calculate summary metrics of the agreement of two FIMs over a given ROI.

### Arguments  


### Input  
- **agreement_path**
  - Path to an agreement gpkg containing a multipoint geometry with the attributes over which the metrics will be calculated.

### Output  
- **metrics_path**
  - The output will be a json file containing the metrics the user requested. `metrics_calculator.yml` lists a small subset of possible metrics.
---
