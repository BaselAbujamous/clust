logfile = ''
tmpfile = 'tmp'
object_label_upper = 'Gene'
object_label_lower = 'gene'
outputwidth = 73
version = 'v1.18.0'
email = 'baselabujamous@gmail.com'
maxgenesinsetforpdist = 10000
print_to_log_file = True
print_to_console = True


def set_logfile(val):
    global logfile
    logfile = val


def set_tmpfile(val):
    global tmpfile
    tmpfile = val


def set_object_label_upper(val):
    global object_label_upper
    object_label_upper = val


def set_object_label_lower(val):
    global object_label_lower
    object_label_lower = val


def set_outputwith(val):
    global outputwidth
    outputwidth = val


def set_version(val):
    global version
    version = val


def set_print_to_log_file(val):
    global print_to_log_file
    print_to_log_file = val


def set_print_to_console(val):
    global print_to_console
    print_to_console = val
