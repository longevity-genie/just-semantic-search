from pyrsistent import PClass, field, pset
from just_semantic_search.embeddings import *
from just_semantic_search.utils.tokens import *
from pathlib import Path
#from just_semantic_search.utils import RenderingFileDestination


import typer
import os
from just_semantic_search.meili.rag import *
from just_semantic_search.meili.rag import *
import time
from pathlib import Path
from eliot import FileDestination
import sys
from datetime import datetime
from eliottree import tasks_from_iterable, render_tasks
from eliot import Logger
from eliot.json import _dumps_bytes, _dumps_unicode, json_default
import json

from eliot._output import *
import uuid
from enum import IntEnum
from io import TextIOBase, IOBase


class LogLevel(IntEnum):
    """Enumeration of log levels with their corresponding numeric values"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class RenderingFileDestination(FileDestination):
    """
    A FileDestination that also renders messages in a human-readable format.
    
    Args:
        json_file: File object for JSON log output (can be Path or TextIO)
        rendered_file: File object for human-readable log output (can be Path or TextIO)
        encoder: Optional custom JSON encoder
    """
    rendered_file = field(mandatory=True)

    def __new__(cls, json_file, rendered_file, encoder=None):
        # Handle both Path/string and TextIO objects for json_file
        if isinstance(json_file, (str, Path)):
            json_file = open(json_file, 'a')
        
        # Handle both Path/string and TextIO objects for rendered_file
        if isinstance(rendered_file, (str, Path)):
            rendered_file = open(rendered_file, 'a')

        return PClass.__new__(cls,
                            file=json_file,
                            rendered_file=rendered_file,
                            _dumps=_dumps_unicode if isinstance(json_file, (IOBase, TextIOBase)) else _dumps_bytes,
                            _linebreak="\n" if isinstance(json_file, (IOBase, TextIOBase)) else b"\n",
                            _json_default=json_default)

    def __call__(self, message):
        # First let parent class handle the JSON file writing
        super().__call__(message)
        
        try:
            # Convert the message to tasks and render them
            tasks = list(tasks_from_iterable([message]))
            render_tasks(self.rendered_file.write, tasks, colorize=False, human_readable=True)
            self.rendered_file.flush()
        except Exception as e:
            self.rendered_file.write(f"Error rendering message: {str(e)}\n")
            self.rendered_file.flush()

def to_nice_stdout(output_file: Optional[Path] = None, encoder=None, json_default=json_default):
    """Configure Eliot logging with improved rendering to stdout
    
    Args:
        output_file (Optional[Path]): Path to the JSON log file. If None, creates a temporary file
        encoder: Optional custom JSON encoder
        json_default: JSON serialization function for unknown types
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = Path("/tmp/just_semantic_search_logs")
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_file = temp_dir / f"log_{timestamp}_{uuid.uuid4().hex[:8]}.json"
    
    destination = RenderingFileDestination(
        json_file=output_file,
        rendered_file=sys.stdout
    )
    Logger._destinations.add(destination)


def to_nice_file(output_file: Path, rendered_file: Path, encoder=None, json_default=json_default):
    """Configure Eliot logging with improved rendering
    
    Args:
        output_file (Path): Path to the JSON log file
        rendered_file (Path): Path to the human-readable rendered log file
        encoder: Optional custom JSON encoder
        json_default: JSON serialization function for unknown types
    """
    destination = RenderingFileDestination(
        json_file=output_file,
        rendered_file=rendered_file
    )
    Logger._destinations.add(destination)
    # stdout: TextIO 