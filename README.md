# elevenlabs_stt_ha

## How to use
Add the following to the `configuration.yaml` file:

```
stt:
  - platform: elevenlabs_stt
    api_key: YOUR_ELEVENLABS_API_KEY
    # Optional parameters:
    # model: scribe_v1
    # language: en  # or another value from the SUPPORTED_LANGUAGES list
    # diarize: true  # to distinguish speakers
    # tag_audio_events: true  # to detect audio events
    # api_url: https://api.elevenlabs.io/v1  # you can change the API URL
```