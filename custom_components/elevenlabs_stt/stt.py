from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterable
import wave
import io
import json

from elevenlabs.client import ElevenLabs
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
CONF_LANGUAGE = "language_code"
CONF_DIARIZE = "diarize"
CONF_TAG_AUDIO_EVENTS = "tag_audio_events"

DEFAULT_MODEL = "scribe_v1"
DEFAULT_LANGUAGE = None  # Auto-detect
DEFAULT_DIARIZE = False
DEFAULT_TAG_AUDIO_EVENTS = False

SUPPORTED_MODELS = [
    "scribe_v1",
]

# ISO 639-2 language codes
SUPPORTED_LANGUAGES = [
    "auto",  # Auto-detect language
    "en",  # English
    "fr",  # French
    "de",  # German
    "it",  # Italian
    "es",  # Spanish
    "pt",  # Portuguese
    "pl",  # Polish
    "nl",  # Dutch
    "ru",  # Russian
    "ja",  # Japanese
    "zh",  # Mandarin Chinese
    "ko",  # Korean
    "hi",  # Hindi
    "ar",  # Arabic
    "tr",  # Turkish
    "sv",  # Swedish
    "fi",  # Finnish
    "da",  # Danish
    "no",  # Norwegian
    "cs",  # Czech
    "hu",  # Hungarian
    "el",  # Greek
    "ro",  # Romanian
    "bg",  # Bulgarian
    "hr",  # Croatian
    "uk",  # Ukrainian
    "he",  # Hebrew
    "ca",  # Catalan
    "sk",  # Slovak
    "lt",  # Lithuanian
    "et",  # Estonian
    "lv",  # Latvian
    "sl",  # Slovenian
    "cy",  # Welsh
    "id",  # Indonesian
    "ms",  # Malay
    "vi",  # Vietnamese
    "th",  # Thai
    "bn",  # Bengali
    "ta",  # Tamil
    "te",  # Telugu
    "mr",  # Marathi
    "kn",  # Kannada
    "ml",  # Malayalam
    "gu",  # Gujarati
    "pa",  # Punjabi
    "ur",  # Urdu
    "fa",  # Persian
    "sw",  # Swahili
]

MODEL_SCHEMA = vol.In(SUPPORTED_MODELS)

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend({
    vol.Required(CONF_API_KEY): cv.string,
    vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): MODEL_SCHEMA,
    vol.Optional(CONF_LANGUAGE, default=DEFAULT_LANGUAGE): vol.Any(None, cv.string),
    vol.Optional(CONF_DIARIZE, default=DEFAULT_DIARIZE): cv.boolean,
    vol.Optional(CONF_TAG_AUDIO_EVENTS, default=DEFAULT_TAG_AUDIO_EVENTS): cv.boolean,
})


async def async_get_engine(hass, config, discovery_info=None):
    """Set up the ElevenLabs STT component."""
    api_key = config[CONF_API_KEY]
    model = config.get(CONF_MODEL, DEFAULT_MODEL)
    language_code = config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
    diarize = config.get(CONF_DIARIZE, DEFAULT_DIARIZE)
    tag_audio_events = config.get(CONF_TAG_AUDIO_EVENTS, DEFAULT_TAG_AUDIO_EVENTS)

    return ElevenLabsSTTProvider(hass, api_key, model, language_code, diarize, tag_audio_events)


class ElevenLabsSTTProvider(Provider):
    """The ElevenLabs STT provider."""

    def __init__(self, hass, api_key, model, language_code, diarize, tag_audio_events) -> None:
        """Init ElevenLabs STT service."""
        self.hass = hass
        self.name = "ElevenLabs STT"

        self._api_key = api_key
        self._model = model
        self._language_code = language_code
        self._diarize = diarize
        self._tag_audio_events = tag_audio_events

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV, AudioFormats.MP3, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM, AudioCodecs.MP3, AudioCodecs.OPUS]

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
            AudioSampleRates.SAMPLERATE_22050,
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
        language_code = metadata.language if metadata.language else self._language_code

        # Map Home Assistant language codes to ISO 639-2 if needed
        # This is a simplified mapping - expand as needed
        language_map = {
            "en": "eng",
            "fr": "fra",
            "de": "deu",
            "it": "ita",
            "es": "spa",
            "pl": "pol",
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

        if language_code and len(language_code) == 2 and language_code in language_map:
            language_code = language_map[language_code]

        def job():
            client = ElevenLabs(api_key=self._api_key)

            try:
                transcription = client.speech_to_text.convert(
                    file=audio_stream,
                    model_id=self._model,
                    language_code=language_code,
                    diarize=self._diarize,
                    tag_audio_events=self._tag_audio_events,
                )

                return {"text": transcription.text}
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