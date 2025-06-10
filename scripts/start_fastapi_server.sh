#!/bin/bash
#uvicorn fast:app --reload # For local deployment
uvicorn fast:app --host 0.0.0.0 --port 8000
