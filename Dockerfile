# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- NEW STEP: Download the spaCy model using its own tool ---
RUN python -m spacy download en_core_web_sm

# Copy the rest of your application's code into the container
COPY . .

# Download the necessary NLTK data into the container
RUN python -m nltk.downloader stopwords punkt averaged_perceptron_tagger maxent_ne_chunker words vader_lexicon

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]