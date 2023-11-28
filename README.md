# Music Generation System

This project is a music generation system that can create original songs based on some input parameters, such as genre, tempo, key, length, etc. The system uses deep learning models to generate the melody, the lyrics, and the harmony of the songs, and outputs them in various formats, such as MIDI, waveform, score, and lyrics. The system also provides feedback and suggestions to improve the songs, and allows the user to rate the songs based on some criteria, such as musicality, creativity, and user satisfaction.

## Installation

To install this project, you need to have Python 3 and pip installed on your system. You can check the installation guide for your operating system [here].

You also need to create a virtual environment for your project using the `venv` module. You can follow the instructions [here] to create and activate a virtual environment.

Then, you need to install Flask and other required libraries using the `pip install -r requirements.txt` command in your project directory. The `requirements.txt` file contains the list of libraries and their versions that are needed for this project.

## Usage

To run this project, you need to set the `FLASK_APP` environment variable to the name of your main Python file, such as `app.py`. You can use the `export FLASK_APP=app.py` command on Linux or macOS, or the `set FLASK_APP=app.py` command on Windows.

You also need to set the `FLASK_ENV` environment variable to `development` to enable the debug mode and the auto-reload feature. You can use the `export FLASK_ENV=development` command on Linux or macOS, or the `set FLASK_ENV=development` command on Windows.

Then, you need to run the Flask application using the `flask run` command. You should see a message like this:

Serving Flask app ‘app.py’ (lazy loading)
Environment: development
Debug mode: on
Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
Restarting with stat
Debugger is active!
Debugger PIN: 123-456-789

Finally, you need to open your web browser and go to the URL http://127.0.0.1:5000/ to see the home page of your web application. You can also use the debugger PIN to access the interactive debugger if you encounter any errors.

## Features

The Flask web application consists of the following features:

- A home page that displays the title, the description, and the instructions of the music generation system, as well as a button to start the song generation process
- A song generation page that allows the user to specify some parameters for the song generation, such as genre, tempo, key, length, etc., and displays a loading animation while the song is being generated
- A song display page that outputs and displays the generated song in various formats, such as MIDI, waveform, score, and lyrics, as well as plays the song in the browser, and provides some feedback and suggestions for improving the song
- A feedback page that allows the user to rate the song based on some criteria, such as musicality, creativity, and user satisfaction, and provide some comments or suggestions for the music generation system

## License

This project is licensed under the MIT License - see the [LICENSE] file for details.
