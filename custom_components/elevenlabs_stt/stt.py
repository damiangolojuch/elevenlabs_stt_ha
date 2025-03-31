from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterable
import wave
import io
import json
import requests

import async_timeout
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    Provider,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)

import homeassistant.helpers.config_validation as cv
import voluptuous as vol

_LOGGER = logging.getLogger(__name__)

CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_LANGUAGE = "language"
CONF_DIARIZE = "diarize"
CONF_TAG_AUDIO_EVENTS = "tag_audio_events"
CONF_API_URL = "api_url"

DEFAULT_API_URL = "https://api.elevenlabs.io/v1"
DEFAULT_MODEL = "scribe_v1"
DEFAULT_LANGUAGE = "auto"
DEFAULT_DIARIZE = False
DEFAULT_TAG_AUDIO_EVENTS = False

SUPPORTED_MODELS = [
    "scribe_v1",
]

SUPPORTED_LANGUAGES = [
    "auto",  # Auto-detect language
    "af",  # Afrikaans
    "ar",  # Arabic
    "hy",  # Armenian
    "az",  # Azerbaijani
    "be",  # Belarusian
    "bs",  # Bosnian
    "bg",  # Bulgarian
    "ca",  # Catalan
    "zh",  # Chinese
    "hr",  # Croatian
    "cs",  # Czech
    "da",  # Danish
    "nl",  # Dutch
    "en",  # English
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "gl",  # Galician
    "de",  # German
    "el",  # Greek
    "he",  # Hebrew
    "hi",  # Hindi
    "hu",  # Hungarian
    "is",  # Icelandic
    "id",  # Indonesian
    "it",  # Italian
    "ja",  # Japanese
    "kn",  # Kannada
    "kk",  # Kazakh
    "ko",  # Korean
    "lv",  # Latvian
    "lt",  # Lithuanian
    "mk",  # Macedonian
    "ms",  # Malay
    "mr",  # Marathi
    "mi",  # Maori
    "ne",  # Nepali
    "no",  # Norwegian
    "fa",  # Persian
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "ru",  # Russian
    "sr",  # Serbian
    "sk",  # Slovak
    "sl",  # Slovenian
    "es",  # Spanish
    "sw",  # Swahili
    "sv",  # Swedish
    "tl",  # Tagalog
    "ta",  # Tamil
    "th",  # Thai
    "tr",  # Turkish
    "uk",  # Ukrainian
    "ur",  # Urdu
    "vi",  # Vietnamese
    "cy",  # Welsh
]

MODEL_SCHEMA = vol.In(SUPPORTED_MODELS)

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend({
    vol.Required(CONF_API_KEY): cv.string,
    vol.Optional(CONF_API_URL, default=DEFAULT_API_URL): cv.string,
    vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): MODEL_SCHEMA,
    vol.Optional(CONF_LANGUAGE, default=DEFAULT_LANGUAGE): cv.string,
    vol.Optional(CONF_DIARIZE, default=DEFAULT_DIARIZE): cv.boolean,
    vol.Optional(CONF_TAG_AUDIO_EVENTS, default=DEFAULT_TAG_AUDIO_EVENTS): cv.boolean,
})


async def async_get_engine(hass, config, discovery_info=None):
    """Set up the ElevenLabs STT component."""
    api_key = config[CONF_API_KEY]
    api_url = config.get(CONF_API_URL, DEFAULT_API_URL)
    model = config.get(CONF_MODEL, DEFAULT_MODEL)
    language = config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
    diarize = config.get(CONF_DIARIZE, DEFAULT_DIARIZE)
    tag_audio_events = config.get(CONF_TAG_AUDIO_EVENTS, DEFAULT_TAG_AUDIO_EVENTS)

    return ElevenLabsSTTProvider(hass, api_key, api_url, model, language, diarize, tag_audio_events)


class ElevenLabsSTTProvider(Provider):
    """The ElevenLabs STT provider."""

    def __init__(self, hass, api_key, api_url, model, language, diarize, tag_audio_events) -> None:
        """Init ElevenLabs STT service."""
        self.hass = hass
        self.name = "ElevenLabs STT"

        self._api_key = api_key
        self._api_url = api_url
        self._model = model
        self._language = language
        self._diarize = diarize
        self._tag_audio_events = tag_audio_events

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [
            AudioSampleRates.SAMPLERATE_8000,
            AudioSampleRates.SAMPLERATE_16000,
            AudioSampleRates.SAMPLERATE_22000,
            AudioSampleRates.SAMPLERATE_44100,
            AudioSampleRates.SAMPLERATE_48000,
        ]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO, AudioChannels.CHANNEL_STEREO]

    async def async_process_audio_stream(
            self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an audio stream to STT service."""
        # Collect audio data
        audio_data = b""
        async for chunk in stream:
            audio_data += chunk

        # Prepare the audio file in the correct format
        audio_stream = io.BytesIO()

        with wave.open(audio_stream, 'wb') as wf:
            wf.setnchannels(metadata.channel)
            wf.setsampwidth(metadata.bit_rate // 8)
            wf.setframerate(metadata.sample_rate)
            wf.writeframes(audio_data)

        audio_stream.seek(0)  # Reset file pointer to beginning

        # Determine language parameter
        language_code = metadata.language if metadata.language != "auto" else self._language

        # Map Home Assistant language codes (ISO 639-1) to ElevenLabs language codes (ISO 639-2)
        language_map = {
            "auto": None,  # Auto-detect
            "en": "eng",
            "fr": "fra",
            "de": "deu",
            "it": "ita",
            "es": "spa",
            "pt": "por",
            "pl": "pol",
            "nl": "nld",
            "ru": "rus",
            "ja": "jpn",
            "zh": "cmn",
            "ko": "kor",
            "hi": "hin",
            "ar": "ara",
            "tr": "tur",
            "sv": "swe",
            "fi": "fin",
            "da": "dan",
            "no": "nor",
            "cs": "ces",
            "hu": "hun",
            "el": "ell",
            "ro": "ron",
            "bg": "bul",
            "hr": "hrv",
            "uk": "ukr",
            "he": "heb",
            "ca": "cat",
            "sk": "slk",
            "lt": "lit",
            "et": "est",
            "lv": "lav",
            "sl": "slv",
            "cy": "cym",
            "id": "ind",
            "ms": "msa",
            "vi": "vie",
            "th": "tha",
            "bn": "ben",
            "ta": "tam",
            "te": "tel",
            "mr": "mar",
            "kn": "kan",
            "ml": "mal",
            "gu": "guj",
            "pa": "pan",
            "ur": "urd",
            "fa": "fas",
            "sw": "swh",
        }

        # Convert language code to ElevenLabs format if it exists in our mapping
        elevenlabs_language_code = language_map.get(language_code)

        def job():
            url = f"{self._api_url}/speech-to-text"

            headers = {
                "xi-api-key": self._api_key,
                "Accept": "application/json"
            }

            data = {
                'model_id': self._model,
                'diarize': str(self._diarize).lower(),
                'tag_audio_events': str(self._tag_audio_events).lower(),
            }

            # Only add language if not auto
            if language_code != "auto" and elevenlabs_language_code is not None:
                data['language_code'] = elevenlabs_language_code

            files = {
                'file': ('audio.wav', audio_stream, 'audio/wav'),
            }

            try:
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data  # Przekazujemy dane jako część formularza, nie jako parametry URL
                )

                if response.status_code == 200:
                    result = response.json()
                    return result
                else:
                    _LOGGER.error(
                        "ElevenLabs STT request failed with status %s: %s",
                        response.status_code,
                        response.text,
                    )
                    return None
            except Exception as e:
                _LOGGER.error(
                    "ElevenLabs STT request failed: %s",
                    str(e),
                )
                return None

        async with async_timeout.timeout(30):  # Longer timeout for STT processing
            assert self.hass
            response = await self.hass.async_add_executor_job(job)

            if response and "text" in response:
                return SpeechResult(
                    response["text"],
                    SpeechResultState.SUCCESS,
                )

            return SpeechResult("", SpeechResultState.ERROR)