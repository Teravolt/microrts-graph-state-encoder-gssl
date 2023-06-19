# pylint: skip-file

import logging

class LoggingUtils:

    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT)

    microrts_parser_logger = logging.getLogger("microrts_parser")
    microrts_parser_logger.setLevel(logging.CRITICAL)
