#!/bin/bash

# We are serious this time! If anything fails, we stop!
set -e

# The final, permanent home for our models!
DEST_DIR="/media/scratch/mferguson/legacy_models"
# A temporary, secret lair for our zip files!
TEMP_DIR=$(mktemp -d)

echo "Okay, deep breath! Let's try this again!"
echo "Creating our directories..."
mkdir -p "$DEST_DIR"
mkdir -p "$TEMP_DIR"
echo "✅ Directories are ready and waiting!"

# --- GloVe's Grand Entrance ---
echo "⬇️  Downloading the GloVe zip file to my secret lair: $TEMP_DIR"
kaggle datasets download -d thanakomsn/glove6b300dtxt -p "$TEMP_DIR"
echo "✅ GloVe zip file has been captured!"
echo "📦 Now, let's unpack it DIRECTLY into its REAL home!"
unzip "$TEMP_DIR/glove6b300dtxt.zip" -d "$DEST_DIR"
echo "✨ GloVe is officially home safe and sound!"

# --- Word2Vec's Wonderful Arrival! ---
echo "⬇️  Downloading the Word2Vec zip file to the secret lair..."
kaggle datasets download -d leadbest/googlenewsvectorsnegative300 -p "$TEMP_DIR"
echo "✅ Word2Vec zip file has been secured!"
echo "📦 Unpacking Word2Vec DIRECTLY into its final home!"
unzip "$TEMP_DIR/googlenewsvectorsnegative300.zip" -d "$DEST_DIR"
echo "✨ Word2Vec has officially arrived!"

# --- The Final Cleanup ---
echo "🧹 Now that everyone is home, I'll clean up my secret lair..."
rm -rf "$TEMP_DIR"
echo "✅ All clean!"

echo "🎉 YAY! The data-ghost has been busted! The models should be there now! Let's check!"
# This is the proof! Let's see what's in there!
ls -l "$DEST_DIR"