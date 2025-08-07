from flask import Blueprint, render_template, request
from ..models import *

history_bp = Blueprint('history', __name__)

