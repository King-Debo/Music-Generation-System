# Import the required libraries
import os
import torch
import mido
import librosa
import librosa.display
import matplotlib.pyplot as plt
import midiutil
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from model import Generator
from data import N_GENRES, N_TEMPOS, N_KEYS, N_MODES, GENRE_DICT, TEMPO_DICT, KEY_DICT, MODE_DICT, SAMPLE_RATE, N_MELS, HOP_LENGTH, WINDOW_SIZE

# Define some constants
APP_DIR = "app" # The directory where the app files will be stored
STATIC_DIR = "static" # The directory where the static files will be stored
TEMPLATE_DIR = "templates" # The directory where the template files will be stored
MODEL_DIR = "model" # The directory where the model files will be stored
MODEL_FILE = "model.pth" # The file name for the model file
MIDI_FILE = "song.midi" # The file name for the MIDI file
WAVE_FILE = "song.wav" # The file name for the waveform file
SCORE_FILE = "song.png" # The file name for the score file
LYRICS_FILE = "song.txt" # The file name for the lyrics file
SECRET_KEY = "music" # The secret key for the app

# Create an instance of the Flask class
app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)
# Set the secret key for the app
app.secret_key = SECRET_KEY
# Create an instance of the Generator class
generator = Generator()
# Load the model parameters from the file
generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_FILE)))
# Set the generator to evaluation mode
generator.eval()

# Define a function to handle the home page
@app.route("/")
def home():
  # Render the home page template
  return render_template("home.html")

# Define a function to handle the song generation page
@app.route("/generate", methods=["GET", "POST"])
def generate():
  # Check if the request method is GET
  if request.method == "GET":
    # Render the song generation page template
    return render_template("generate.html")
  # Check if the request method is POST
  elif request.method == "POST":
    # Get the input parameters from the form
    genre = request.form.get("genre")
    tempo = request.form.get("tempo")
    key = request.form.get("key")
    mode = request.form.get("mode")
    length = request.form.get("length")
    # Validate the input parameters
    if genre not in GENRE_DICT or tempo not in TEMPO_DICT or key not in KEY_DICT or mode not in MODE_DICT or not length.isdigit():
      # Flash an error message
      flash("Invalid input parameters. Please try again.")
      # Redirect to the song generation page
      return redirect(url_for("generate"))
    # Convert the input parameters to integers
    genre = GENRE_DICT[genre]
    tempo = TEMPO_DICT[tempo]
    key = KEY_DICT[key]
    mode = MODE_DICT[mode]
    length = int(length)
    # Feed the input parameters to the generator to get the output tensor
    output_tensor = generator(genre, tempo, key, length)
    # Split the output tensor of shape (MAX_LENGTH * N_PITCHES, N_VELOCITIES + N_CHORDS + N_BEATS + N_MELS) into the melody tensor of shape (MAX_LENGTH * N_PITCHES, N_VELOCITIES), the lyrics tensor of shape (MAX_LENGTH * N_PITCHES, VOCAB_SIZE), and the harmony tensor of shape (MAX_LENGTH * N_PITCHES, N_CHORDS + N_BEATS + N_MELS) along the second dimension
    melody_tensor, lyrics_tensor, harmony_tensor = torch.split(output_tensor, [N_VELOCITIES, VOCAB_SIZE, N_CHORDS + N_BEATS + N_MELS], dim=1)
    # Convert the melody tensor to a MIDI file and save it
    tensor_to_midi(melody_tensor, os.path.join(APP_DIR, MIDI_FILE))
    # Convert the melody tensor to a waveform file and save it
    tensor_to_wave(melody_tensor, os.path.join(APP_DIR, WAVE_FILE))
    # Convert the melody tensor and the harmony tensor to a score file and save it
    tensor_to_score(melody_tensor, harmony_tensor, os.path.join(APP_DIR, SCORE_FILE))
    # Convert the lyrics tensor to a lyrics file and save it
    tensor_to_lyrics(lyrics_tensor, os.path.join(APP_DIR, LYRICS_FILE))
    # Render the song display page template
    return render_template("display.html")

# Define a function to handle the song display page
@app.route("/display")
def display():
  # Render the song display page template
  return render_template("display.html")

# Define a function to handle the feedback page
@app.route("/feedback", methods=["GET", "POST"])
def feedback():
  # Check if the request method is GET
  if request.method == "GET":
    # Render the feedback page template
    return render_template("feedback.html")
  # Check if the request method is POST
  elif request.method == "POST":
    # Get the feedback parameters from the form
    musicality = request.form.get("musicality")
    creativity = request.form.get("creativity")
    user_satisfaction = request.form.get("user_satisfaction")
    comments = request.form.get("comments")
    # Validate the feedback parameters
    if musicality not in ["1", "2", "3", "4", "5"] or creativity not in ["1", "2", "3", "4", "5"] or user_satisfaction not in ["1", "2", "3", "4", "5"]:
      # Flash an error message
      flash("Invalid feedback parameters. Please try again.")
      # Redirect to the feedback page
      return redirect(url_for("feedback"))
    # Convert the feedback parameters to integers
    musicality = int(musicality)
    creativity = int(creativity)
    user_satisfaction = int(user_satisfaction)
    # Write the feedback parameters and the comments to the validation file
    with open(os.path.join(APP_DIR, VALIDATION_FILE), "a") as f:
      f.write(f"Musicality: {musicality}, Creativity: {creativity}, User satisfaction: {user_satisfaction}, Comments: {comments}\n")
    # Flash a success message
    flash("Thank you for your feedback. We appreciate your help in improving our music generation system.")
    # Redirect to the home page
    return redirect(url_for("home"))

# Define a function to convert a melody tensor to a MIDI file
def tensor_to_midi(tensor, file_path):
  # Create an instance of the MidiFile class
  midi_file = mido.MidiFile()
  # Create an instance of the MidiTrack class
  midi_track = mido.MidiTrack()
  # Append the midi track to the midi file
  midi_file.tracks.append(midi_track)
  # Set the tempo to 120 beats per minute
  tempo = 120
  # Set the ticks per beat to 480
  ticks_per_beat = 480
  # Convert the tempo to microseconds per beat
  microseconds_per_beat = mido.bpm2tempo(tempo)
  # Convert the tensor to a numpy array
  array = tensor.numpy()
  # Initialize a variable to store the current time in ticks
  current_time = 0
  # Initialize an empty dictionary to store the active notes and their velocities
  active_notes = {}
  # Loop through the rows of the array
  for row in array:
    # Split the row into the velocity vector of length N_VELOCITIES and the rest vector of length N_CHORDS + N_BEATS + N_MELS
    velocity_vector, rest_vector = np.split(row, [N_VELOCITIES])
    # Get the index of the maximum value in the velocity vector
    index = np.argmax(velocity_vector)
    # Check if the index is zero, which means no note is played
    if index == 0:
      # Increment the current time by one tick
      current_time += 1
    else:
      # Get the note and the velocity from the index
      note = index - 1
      velocity = velocity_vector[index]
      # Check if the note is in the active notes dictionary
      if note in active_notes:
        # Get the previous velocity from the active notes dictionary
        previous_velocity = active_notes[note]
        # Check if the velocity is different from the previous velocity
        if velocity != previous_velocity:
          # Create a note off message for the note with the previous velocity and the current time
          note_off_msg = mido.Message("note_off", note=note, velocity=previous_velocity, time=current_time)
          # Append the note off message to the midi track
          midi_track.append(note_off_msg)
          # Create a note on message for the note with the velocity and zero time
          note_on_msg = mido.Message("note_on", note=note, velocity=velocity, time=0)
          # Append the note on message to the midi track
          midi_track.append(note_on_msg)
          # Update the active notes dictionary with the new velocity
          active_notes[note] = velocity
          # Reset the current time to zero
          current_time = 0
      else:
        # Create a note on message for the note with the velocity and the current time
        note_on_msg = mido.Message("note_on", note=note, velocity=velocity, time=current_time)
        # Append the note on message to the midi track
        midi_track.append(note_on_msg)
        # Add the note and the velocity to the active notes dictionary
        active_notes[note] = velocity
        # Reset the current time to zero
        current_time = 0
  # Loop through the active notes dictionary
  for note, velocity in active_notes.items():
    # Create a note off message for the note with the velocity and the current time
    note_off_msg = mido.Message("note_off", note=note, velocity=velocity, time=current_time)
    # Append the note off message to the midi track
    midi_track.append(note_off_msg)
    # Increment the current time by one tick
    current_time += 1
  # Save the midi file to the file path
  midi_file.save(file_path)

# Define a function to convert a melody tensor to a waveform file
def tensor_to_wave(tensor, file_path):
  # Create an instance of the MidiFile class
  midi_file = mido.MidiFile()
  # Create an instance of the MidiTrack class
  midi_track = mido.MidiTrack()
  # Append the midi track to the midi file
  midi_file.tracks.append(midi_track)
  # Set the tempo to 120 beats per minute
  tempo = 120
  # Set the ticks per beat to 480
  ticks_per_beat = 480
  # Convert the tempo to microseconds per beat
  microseconds_per_beat = mido.bpm2tempo(tempo)
  # Convert the tensor to a numpy array
  array = tensor.numpy()
  # Initialize a variable to store the current time in ticks
  current_time = 0
  # Initialize an empty dictionary to store the active notes and their velocities
  active_notes = {}
  # Loop through the rows of the array
  for row in array:
    # Split the row into the velocity vector of length N_VELOCITIES and the rest vector of length N_CHORDS + N_BEATS + N_MELS
    velocity_vector, rest_vector = np.split(row, [N_VELOCITIES])
    # Get the index of the maximum value in the velocity vector
    index = np.argmax(velocity_vector)
    # Check if the index is zero, which means no note is played
    if index == 0:
      # Increment the current time by one tick
      current_time += 1
    else:
      # Get the note and the velocity from the index
      note = index - 1
      velocity = velocity_vector[index]
      # Check if the note is in the active notes dictionary
      if note in active_notes:
        # Get the previous velocity from the active notes dictionary
        previous_velocity = active_notes[note]
        # Check if the velocity is different from the previous velocity
        if velocity != previous_velocity:
          # Create a note off message for the note with the previous velocity and the current time
          note_off_msg = mido.Message("note_off", note=note, velocity=previous_velocity, time=current_time)
          # Append the note off message to the midi track
          midi_track.append(note_off_msg)
          # Create a note on message for the note with the velocity and zero time
          note_on_msg = mido.Message("note_on", note=note, velocity=velocity, time=0)
          # Append the note on message to the midi track
          midi_track.append(note_on_msg)
          # Update the active notes dictionary with the new velocity
          active_notes[note] = velocity
          # Reset the current time to zero
          current_time = 0
      else:
        # Create a note on message for the note with the velocity and the current time
        note_on_msg = mido.Message("note_on", note=note, velocity=velocity, time=current_time)
        # Append the note on message to the midi track
        midi_track.append(note_on_msg)
        # Add the note and the velocity to the active notes dictionary
        active_notes[note] = velocity
        # Reset the current time to zero
        current_time = 0
  # Loop through the active notes dictionary
  for note, velocity in active_notes.items():
    # Create a note off message for the note with the velocity and the current time
    note_off_msg = mido.Message("note_off", note=note, velocity=velocity, time=current_time)
    # Append the note off message to the midi track
    midi_track.append(note_off_msg)
    # Increment the current time by one tick
    current_time += 1
  # Convert the midi file to a waveform array
  wave_array = mido.MidiFile.play(midi_file, meta_messages=True)
  # Convert the waveform array to a waveform file and save it
  librosa.output.write_wav(file_path, wave_array, SAMPLE_RATE)

# Define a function to convert a melody tensor and a harmony tensor to a score file
def tensor_to_score(melody_tensor, harmony_tensor, file_path):
  # Create an instance of the MIDIFile class with one track
  midi_file = midiutil.MIDIFile(1)
  # Set the tempo to 120 beats per minute
  tempo = 120
  # Set the track to 0
  track = 0
  # Set the channel to 0
  channel = 0
  # Set the time to 0
  time = 0
  # Add the tempo to the midi file
  midi_file.addTempo(track, time, tempo)
  # Convert the melody tensor to a numpy array
  melody_array = melody_tensor.numpy()
  # Convert the harmony tensor to a numpy array
  harmony_array = harmony_tensor.numpy()
  # Initialize a variable to store the current time in seconds
  current_time = 0
  # Initialize an empty dictionary to store the active notes and their durations
  active_notes = {}
  # Loop through the rows of the melody array
  for row in melody_array:
    # Split the row into the velocity vector of length N_VELOCITIES and the rest vector of length N_CHORDS + N_BEATS + N_MELS
    velocity_vector, rest_vector = np.split(row, [N_VELOCITIES])
    # Get the index of the maximum value in the velocity vector
    index = np.argmax(velocity_vector)
    # Check if the index is zero, which means no note is played
    if index == 0:
      # Increment the current time by 0.25 seconds
      current_time += 0.25
    else:
      # Get the note and the velocity from the index
      note = index - 1
      velocity = velocity_vector[index]
      # Check if the note is in the active notes dictionary
      if note in active_notes:
        # Get the previous velocity and the start time from the active notes dictionary
        previous_velocity, start_time = active_notes[note]
        # Check if the velocity is different from the previous velocity
        if velocity != previous_velocity:
          # Compute the duration by subtracting the start time from the current time
          duration = current_time - start_time
          # Add the note, the start time, the duration, the channel, and the previous velocity to the midi file
          midi_file.addNote(track, channel, note, start_time, duration, previous_velocity)
          # Update the active notes dictionary with the new velocity and the current time
          active_notes[note] = (velocity, current_time)
      else:
        # Add the note and the velocity and the current time to the active notes dictionary
        active_notes[note] = (velocity, current_time)
  # Loop through the active notes dictionary
  for note, (velocity, start_time) in active_notes.items():
    # Compute the duration by subtracting the start time from the current time
    duration = current_time - start_time
    # Add the note, the start time, the duration, the channel, and the velocity to the midi file
    midi_file.addNote(track, channel, note, start_time, duration, velocity)
  # Initialize a variable to store the current chord in index
  current_chord = 0
  # Initialize a variable to store the current beat in index
  current_beat = 0
  # Loop through the rows of the harmony array
  for row in harmony_array:
    # Split the row into the chord vector of length N_CHORDS, the beat vector of length N_BEATS, and the mel vector of length N_MELS
    chord_vector, beat_vector, mel_vector = np.split(row, [N_CHORDS, N_CHORDS + N_BEATS])
    # Get the index of the maximum value in the chord vector
    chord_index = np.argmax(chord_vector)
    # Get the index of the maximum value in the beat vector
    beat_index = np.argmax(beat_vector)
    # Check if the chord index is different from the current chord
    if chord_index != current_chord:
      # Update the current chord
      current_chord = chord_index
      # Add the chord to the midi file as a text event
      midi_file.addText(track, current_time, f"Chord: {current_chord}")
    # Check if the beat index is different from the current beat
    if beat_index != current_beat:
      # Update the current beat
      current_beat = beat_index
      # Add the beat to the midi file as a text event
      midi_file.addText(track, current_time, f"Beat: {current_beat}")
    # Increment the current time by 0.25 seconds
    current_time += 0.25
  # Write the midi file to a binary file
  with open("temp.midi", "wb") as f:
    midi_file.writeFile(f)
  # Convert the midi file to a score using the music21 library
  score = music21.converter.parse("temp.midi")
  # Plot the score using the matplotlib library
  plt.figure(figsize=(20, 10))
  plt.title("Score")
  music21.plot.plotStream(score, 'pianoRoll')
  # Save the plot to the file path
  plt.savefig(file_path)

# Define a function to convert a lyrics tensor to a lyrics file
def tensor_to_lyrics(tensor, file_path):
  # Create an instance of the GPT2Tokenizer class
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  # Convert the tensor to a numpy array
  array = tensor.numpy()
  # Initialize an empty list to store the tokens
  tokens = []
  # Loop through the rows of the array
  for row in array:
    # Get the index of the maximum value in the row
    index = np.argmax(row)
    # Get the token from the index
    token = tokenizer.decode(index)
    # Append the token to the tokens list
    tokens.append(token)
  # Join the tokens to a string
  string = "".join(tokens)
  # Write the string to the file path
  with open(file_path, "w") as f:
    f.write(string)
