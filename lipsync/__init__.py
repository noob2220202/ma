"""Still-image lip-sync built on the Wav2Lip generator."""


class LipsyncError(Exception):
    """A failure caused by the inputs or the environment rather than by a bug: the message says what to fix."""
