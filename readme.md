# Info

Continuous and Data-Driven Descriptors (CDDD) docker image serving embedding predictions for provided SMILES. The method implementation is found [here](https://github.com/jrwnter/cddd). 
Due to numerous dependency errors, and problems running in a local environment, this image provides self-contained environment to transform a list of SMILES into 512-dimensional embeddings.

# Install & serve

Build docker image
`docker build -t cddd .`

Serve in a local environment (by default the fastAPI is exposed on port `80`).
`docker run --rm -p 80:80  cddd`

# Run predictions 
Use the example file provided.

For unknown smiles model returns `nan`s.

## Curl

```bash
curl -X POST http://localhost:80/predict \
-d @example_batches.json \
-H "Content-Type: application/json"

```