# VisionBridge - Eye Gaze Communication System

VisionBridge is an assistive technology application designed to help ALS patients communicate using eye gaze tracking. The system provides multiple communication modes including response selection, text spelling, voice interaction, and emergency assistance features.

## 🎯 Key Features

- **Eye Gaze Tracking**: Advanced calibration system for precise gaze detection
- **Multiple Communication Modes**:
  - Response selection from AI-generated options
  - Letter-by-letter spelling interface
  - Voice recording and transcription
  - Scheduled question reminders
- **AI Integration**: Uses OpenAI GPT and semantic similarity for intelligent response generation
- **Voice Synthesis**: ElevenLabs voice cloning for personalized speech output
- **Emergency Support**: Automated email alerts to caregivers
- **Machine Learning**: Predictive models for improved gaze accuracy

## 🚀 Quick Start

### Prerequisites

- **Operating System**: Windows 11 (tested on Lenovo Thinkpad E15 Gen 3, 16GB RAM)
- **Python**: 3.8 or higher
- **Webcam**: Required for eye tracking
- **Microphone**: Required for voice input

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Visionbridge-Published-Code-POC
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root with the following variables:
   ```env
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   APP_PASSWORD=your_gmail_app_password
   ELEVEN_API_KEY=your_elevenlabs_api_key
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_FINE_TUNED_API_KEY=your_openai_fine_tuned_key
   ```

   **Required API Keys:**
   - **Supabase**: For data storage and retrieval ([Get API keys](https://supabase.com))
   - **ElevenLabs**: For voice synthesis ([Get API key](https://elevenlabs.io))
   - **OpenAI**: For AI response generation ([Get API key](https://openai.com))
   - **Gmail App Password**: For emergency email notifications ([Setup guide](https://support.google.com/accounts/answer/185833))

4. **Run the application**
   ```bash
   python main.py
   ```

## 📖 How to Use

### Initial Setup

1. **Camera Positioning**: Position your webcam at eye level for optimal tracking
2. **Lighting**: Ensure good lighting on your face
3. **Calibration**: Follow the on-screen calibration process:
   - Look up, center, and down when prompted
   - Look at the calibration points that appear on screen
   - The system will learn your eye movement patterns

### Communication Modes

#### 1. Response Selection Mode
- Ask a question using voice (press Space bar) or through the scheduling interface
- The system generates 4 response options using AI
- Look at the screen region containing your desired response
- Hold your gaze for 3+ seconds to select

#### 2. Spelling Mode
- Look at the top-left corner to enter spelling mode
- Letters are divided into quadrants on the screen
- Look at the quadrant containing your desired letter
- Continue narrowing down until you select individual letters
- Build words letter by letter

#### 3. Voice Input
- Press the Space bar to activate voice recording
- Speak your question clearly
- The system will transcribe and generate response options

#### 4. Scheduled Questions
- Use the GUI interface to schedule recurring questions
- Set specific times or countdown timers
- Questions can repeat daily for routine check-ins

### Screen Regions

The interface is divided into regions for gaze-based interaction:
- **Top**: Response option 1
- **Left**: Response option 2  
- **Bottom**: Response option 3
- **Right**: Response option 4
- **Top-left corner**: Enter spelling mode
- **Top-right corner**: Emergency help (sends alert email)

### Keyboard Controls

- **Space**: Start voice recording
- **Q**: Quit application

## 🔧 Configuration

### Emergency Contact Setup
Configure emergency email notifications by setting the recipient email in the application. When the top-right corner is selected, an alert email is automatically sent to caregivers.

### Voice Customization
The system supports ElevenLabs voice cloning for personalized speech output. Configure your preferred voice through the ElevenLabs API settings.

### AI Response Tuning
Adjust response generation parameters in the code:
- `similarity_threshold`: Controls relevance of database responses (default: 0.55)
- `diversity_threshold`: Ensures response variety (default: 0.7)
- `top_n`: Number of response options (default: 4)

## 🛠️ Troubleshooting

### Common Issues

**Camera not detected**
- Ensure webcam is connected and not used by other applications
- Check camera permissions in Windows settings

**Poor eye tracking accuracy**
- Recalibrate the system
- Improve lighting conditions
- Ensure camera is at eye level
- Minimize head movement during use

**API connection errors**
- Verify all environment variables are set correctly
- Check internet connection
- Confirm API key validity and quotas

**Voice recognition issues**
- Speak clearly and at moderate pace
- Ensure microphone is working and not muted
- Check for background noise interference

## 📞 Support

For technical support or questions, contact: kevinx8017@gmail.com

## ⚠️ Important Disclaimer

This software is provided as a proof-of-concept for research and educational purposes only. It is not a certified medical device and should not be relied upon for diagnosis, treatment, or critical communication in medical or emergency situations. Use at your own risk. The author assumes no responsibility for any consequences arising from the use of this software.

## 🔒 Privacy & Data

- Voice recordings are processed locally using Whisper
- Question/response data is stored in your configured Supabase database
- No personal data is shared with third parties beyond the configured API services
- Emergency emails are sent only when explicitly triggered by the user
