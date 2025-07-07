#!/bin/bash
# Streamlit Cloud startup script

echo "Starting NFL Kicker Leaderboard app..."

# Copy leaderboard files if they exist
if [ -d "output" ]; then
    echo "Copying leaderboard files..."
    python copy_leaderboards.py
fi

# Start the Streamlit app
echo "Starting Streamlit..."
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
