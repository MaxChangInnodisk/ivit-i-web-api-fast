# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


class ErrorWrapper(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class InvalidError(ErrorWrapper):
    pass


class InvalidUidError(ErrorWrapper):
    pass
