## Input and outputs
All data inputs and outputs from containers should be written to and read from files. IO should always use the python [smart_open library](https://pypi.org/project/smart-open/) 

## Entrypoint script arguments

### names and abbreviations

* If an argument is a path then full argument name should have “path” at the end.
* If two or more jobs accept the same argument then the argument name and abbreviation should be the same across all the jobs. 
* If an argument has the same name across jobs then it should also have the same abbreviation.

## Logging

### Log format
All log entries will be a single JSON object per line. The log entries should the following keys:

* timestamp
* level
* tags
  * job_id
* message

An example log entry might be:

```
{
  "timestamp": "2025-04-14T16:10:15.543210Z",
  "level": "INFO",
  "tags": {
    "job_id": "hand_inundator",
    "other_context": "<value>"
    },
  "message": "Began inundate job for branch 0",
}
```

### tags

The tags key is a way that jobs can log additional context about a job run that could be useful for querying across job runs or across jobs. Its value is always a json object whose keys can only contain strings or integers. That is there shouldn't be any further nested objects or lists inside the tags objects. 

### Log levels

Below are guidelines for what to include at each log level:

* Debug: Log fine grained detail about job execution here. Would typically be used for debugging.
* Info: Log messages at this level that track job progression at a high level.
* Warning: Log non-breaking errors here that indicate an out of the ordinary exception occured that was handled.
* Error: Log errors that result in a run failure here.
* Success: A message recording a successful run. Success log messages will also contain keys that correspond to the output paths of a given job so that a successful data write with pathing is recorded.

### Results

When a job runs successfully then the last message should look like this:

```
{
  "timestamp": "2025-04-14T16:10:15.543210Z",
  "level": "SUCCESS",
  "job_id": "hand_inundator",
  "tags": {
    "job_id": "hand_inundator",
    "other_context": "<value>"
    },
  "message": {
    "output_type1": "s3://fimc-data/path/to/filetype1",
    "output_type2": "s3://fimc-data/path/to/file2"
  }
}
```

The message of a success log is a json object whose keys are strings containing filepaths or lists of strings containing filepaths.

### Error logging

If an error message is logged, **it should always be the last message logged by the job**. Error messages will follow a consistent format across jobs. Here is an example:

```
{
  "timestamp": "2025-04-14T16:10:15.543210Z",
  "level": "ERROR",
  "job_id": "hand_inundator",
  "tags": {
    "job_id": "hand_inundator",
    "other_context": "<value>"
    },
  "message": "Inundate job run failed: {fatal error message here}",
}
```

### Logging libraries

The logging module from the python standard library will be used to log messages to stderr. The library python-json-logger will be used to create a formatting object to format the log messages.
