# First stage: Build and cache dependencies
FROM python:3.12.4 as python_cache

# Set up the virtual environment directory
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy only the requirements file to take advantage of Docker layer caching
COPY ./app/requirements.txt /app/requirements.txt

# Create the virtual environment and install dependencies
WORKDIR /app
RUN python -m venv $VIRTUAL_ENV && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Second stage: Build the final image
FROM python:3.12.4

# Copy the cached virtual environment from the first stage
COPY --from=python_cache /opt/venv /opt/venv

# Set up the virtual environment path
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Do not cache this line
RUN echo "Build date: $(date)" > /dev/null

# Copy the application code
COPY ./app /app

# Set the working directory
WORKDIR /app

# Set the entry point to run the Streamlit app
ENTRYPOINT [ "python", "-m", "streamlit", "run", "/app/main.py" ]
