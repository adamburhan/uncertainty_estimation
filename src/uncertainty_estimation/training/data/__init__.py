"""Training-specific data contracts and loaders.

Unlike the sequence-oriented loaders in ``uncertainty_estimation.data``, this
package is reserved for batch-friendly training inputs and metadata tailored to
the stereo self-supervised learning pipeline.
"""

from .common import StereoFrameMetadata
from .batch import StereoTrainingBatch
