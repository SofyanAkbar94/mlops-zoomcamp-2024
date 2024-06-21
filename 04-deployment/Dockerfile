# Use the provided base image
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies via pipenv
RUN pipenv install --deploy --ignore-pipfile

# Copy the script to the container
COPY starter.py .

# Set the entrypoint for the container
ENTRYPOINT ["pipenv", "run", "python", "starter.py"]
