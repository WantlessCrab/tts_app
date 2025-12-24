#!/bin/bash

# Function: Opens a new independent window for each service
launch_service() {
    TITLE=$1
    COMPOSE_FILE=$2
    
    echo "🚀 Launching $TITLE..."
    
    # This magic command tells Windows to open a new WSL window, 
    # navigate to your current folder, and run the docker command.
    cmd.exe /c start "$TITLE" wsl.exe bash -c "cd \"$PWD\"; echo '--- $TITLE ---'; docker compose -f $COMPOSE_FILE up; exec bash"
}

# --- THE SEQUENCE ---

# 1. TTS Service
launch_service "TTS Engine" "docker-compose.tts.py310.yml"
sleep 1

# 2. File Server
launch_service "File Server" "docker-compose.fileserver.yml"
sleep 1

# 3. PDF Processor
launch_service "PDF Processor" "docker-compose.pdf.yml"
sleep 1

# 4. API Gateway
launch_service "Gateway" "docker-compose.gateway.yml"
sleep 1

# 5. (Optional) The Relay Bridge we just fixed
# I included this because you likely want the LLM bridge running too!
launch_service "LLM Relay" "docker-compose.relay.yml"

echo "✅ All launch commands issued."

chmod: cannot access 'start_services.sh': No such file or directory
