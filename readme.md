# Disclaimer 

**Note: this repository is a work in progress. The code is being actively developed and the intended functionality is only partially implemented. These capabilities are scheduled for delivery to OWP in PI-7.**

# About

This repository contains code and tests for a set of containerized flood inundation map (FIM) evaluation jobs that are meant to be composed together to form a FIM evaluation workflow. Each job can be developed and run independently of other jobs or used in conjunction with a job orchestrator to run evaluations at scale. The intended target for the orchestrator is HashiCorp Nomad and the jobs have been designed to make them easy to run as parameterized jobs on a Nomad cluster.  

A more thorough description of the inputs and outputs of each job as well as the intended behavior of a job can be found in the jobs' [interfaces](/interfaces/interfaces.md) descriptions. A job interface is a formal specification of a job's inputs, outputs, and arguments specified using [json-schema](https://json-schema.org/). At the moment the interface yaml files serve as a guide for developers when (re)implementing jobs and for understanding the possible ways that jobs can interact through their inputs and outputs. In the future they could also be used to validate the data accepted and produced by each job.

We also provide another document in the interfaces directory listing [job conventions](/interfaces/job_conventions.md) to use when implementing jobs. These are conventions developers are expected to follow when implementing jobs and include implementation guidelines on how inputs and outputs should be handled by the job, rules for job entrypoint argument names, and job logging.

# Setup

The jobs are currently all designed to be run using the same Docker image. We provide a Dockerfile for this image as well as two Docker compose files. The first Docker compose file contains services for running the jobs in an interactive shell for development and debugging. The second Docker compose file specifies services that run a jobs test suite on startup.  

## Configuring environment

Depending on which job is being run, some configuration of the container's environment may be necessary. Currently, the most important environment variables are the credentials that jobs need to read and write data to AWS S3 buckets.

The docker compose services follow the following policy for environment variables:

- variables that need to be dynamically updated (like AWS credentials with a fixed, short expiration date) will be referenced in the "environment" block of the docker compose service definition and will be read from the user's local shell environment. Currently we are accessing an S3 bucket that requires credentials. To populate your Docker containers with these credentials you would put your AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN in your host computer's bash shell's environment.
- Fixed variables will be referenced from a .env file in an env_file block. An example .env file is located at [example.env](example.env) 

For the hand_inundator and fim_mosaicker jobs the GDAL_CACHEMAX environment variable also needs to be set. This variable controls the amount of memory that the hand_inundator and fim_mosaicker jobs use for raster processing.

## Using Docker Compose dev services to interactively run and debug jobs

Here is an example for how to enter into an interactive shell for the fim_mosaicker job:

```
# start all Docker Compose dev services
docker compose up -d

# enter into a shell for developing fim_mosaicker
docker compose exec mosaic-dev bash
```

You only need to run the `docker compose up -d` command once and then after that you can enter into shells for any job there is a service for. There are currently Docker Compose dev services for fim_mosaicker and hand_inundator. The hand_inundator service is called 'inundate-dev'. To develop hand_inundator you would run `docker compose exec inundate-dev bash` instead of the Docker Compose exec command in the original example.

Once you have entered the shell for a job container, the container's '/app' directory will contain code, tests, and mock data for that job. The bash shell can be exited by entering the `exit` command from bash.

Once you are done developing a job all the dev services can be shutdown with:

```
docker compose down
```

## Running tests

Tests for each job can also be run as Docker Compose services. Typically you would use this command:

```
docker compose -f docker-compose-tests.yml run inundate-test
```

The `inundate-test` portion of the command above will change depending on which job you are testing. For example, to test fim_mosaicker you would replace `inundate-test` with `mosaic-test`.

Each docker service in the tests docker compose file runs tests for a different job.

# License

