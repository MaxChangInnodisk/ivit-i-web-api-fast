# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import common

common.init_ivit_env()

from ivit_i.common import Metric, handle_exception, simple_exception
from ivit_i.common.app import iAPP_CLS, iAPP_HANDLER, iAPP_OBJ, iAPP_SEG
from ivit_i.core.models import iModel
from ivit_i.io import Displayer, RtspWrapper, SourceV2
from ivit_i.utils import iDevice

keep = [
    SourceV2,
    Displayer,
    RtspWrapper,
    iAPP_HANDLER,
    iAPP_CLS,
    iAPP_OBJ,
    iAPP_SEG,
    iModel,
    Metric,
    iDevice,
    simple_exception,
    handle_exception,
]
