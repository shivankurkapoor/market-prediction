"""
This file contains utility functions for the project

If you want to directly analyze the http response, use simple_http_request
Otherwise, if you want directly the responding data, use http_request
Simplfy as simple_http_request to return the raw http response
and http_request to return the processed data
"""

import logging
import urllib
import urllib2
import pandas
import json
from common.consts import *

logger = logging.getLogger(__name__)


def http_request(url=STR_UNDEFINED,
                 params='',
                 method='GET',
                 url2='',
                 ishttps=False,
                 headers=None,
                 port=443):
    """
    This is the method for requesting simple http request, the raw version
    :param url:
    :param method:
    :param body: str
    :param headers: dict, includes all header fields
    :param url2: This is the sub URL address
    :param ishttps: true or false
    :return: status code, data from the response object
    """
    assert url != STR_UNDEFINED
    assert method in LIST_HTTPMETHODS
    assert isinstance(headers, dict) or headers is None
    assert isinstance(params, str) or isinstance(params, dict)

    headers = headers if headers else {}
    req = None
    data = None
    status = INT_ERROR_GENERAL
    if params and isinstance(params, dict):
        params = urllib.urlencode(params)

    url = ('https://%s:%s%s') % (url, port, url2) \
        if ishttps else ('http://%s:%s%s') % (url, port, url2)

    try:
        if method == 'GET':
            if params:
                req = urllib2.Request(url + '?' + params, None, headers)
            else:
                req = urllib2.Request(url, None, headers)

        elif method == 'POST':
            if params:
                req = urllib2.Request(url, params, headers)
            else:
                req = urllib2.Request(url, None, headers)
    except all as e:
        print 'Error in HTTP %s Request: %s' % (method, e)
        raise

    try:
        res = urllib2.urlopen(req)
        if res.code == 200 or res.code == 300:
            data = res.read()
            status = INT_OK
            logger.info('Successfully retrieved response from %s' % (res.url))
    except urllib2.URLError as e:
        print('Error in get response from HTTP %s Request: %s' % (method, e))
        raise
    return status, data
