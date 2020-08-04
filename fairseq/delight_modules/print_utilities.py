# ============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
# ============================================
import time

text_colors = {
               'logs': '\033[34m', # 033 is the escape code and 34 is the color code
               'info': '\033[32m',
               'warning': '\033[33m',
               'error': '\033[31m',
               'bold': '\033[1m',
               'end_color': '\033[0m'
               }


def get_curr_time_stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_error_message(message):
    time_stamp = get_curr_time_stamp()
    error_str = text_colors['error'] + text_colors['bold'] + 'ERROR  ' + text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, error_str, message))
    print('{} - {} - {}'.format(time_stamp, error_str, 'Exiting!!!'))
    exit(-1)


def print_log_message(message):
    time_stamp = get_curr_time_stamp()
    log_str = text_colors['logs'] + text_colors['bold'] + 'LOGS   ' + text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, log_str, message))


def print_warning_message(message):
    time_stamp = get_curr_time_stamp()
    warn_str = text_colors['warning'] + text_colors['bold'] + 'WARNING' + text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, warn_str, message))


def print_info_message(message):
    time_stamp = get_curr_time_stamp()
    info_str = text_colors['info'] + text_colors['bold'] + 'INFO   ' + text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, info_str, message))


def print_dash_line():
    print(text_colors['error'] + text_colors['bold'] + '=' * 100 + text_colors['end_color'])

def print_header(header):
    print_dash_line()
    print(text_colors['info'] + text_colors['bold'] + '=' * 50 + str(header) + text_colors['end_color'])
    print_dash_line()

def print_header_minor(header):
    print(text_colors['warning'] + text_colors['bold'] + '=' * 25 + str(header) + text_colors['end_color'])


if __name__ == '__main__':
    print_log_message('Testing')
    print_warning_message('Testing')
    print_info_message('Testing')
    print_error_message('Testing')
