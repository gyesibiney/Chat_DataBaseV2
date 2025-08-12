FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


# Copy the FastAPI app code to the container
COPY app_fastapi.py .

# Expose the port the FastAPI app will run on
EXPOSE 7860

# The CMD instruction specifies the command to run when the container starts
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "7860"]