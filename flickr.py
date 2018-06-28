from flickrapi import FlickrAPI
from pprint import pprint
import urllib
import time
import os
import sys

FLICKR_PUBLIC = 'a089b9cee66f9387560f6acc591f5795'
FLICKR_SECRET = os.environ['FLICKR_SECRET']

flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
extras='url_sq,url_t,url_s,url_q,url_m,url_n,url_z,url_c,url_l,url_o'

category = sys.argv[1]
print category
c = 'flickr_' + category
if not os.path.exists(c):
	os.makedirs(c)

pages = 10
page = 1
inc = 0
limit = 150000
while page <= pages:
	cats = flickr.photos.search(text=category.replace('_', ' '), per_page=500, sort='relevance', extras=extras, page=page)
	photos = cats['photos']
	pages = photos['pages']
	for p in photos['photo']:
		if 'url_z' in p:
			d = '%s/%02d' % (c, inc // 10000)
			if inc % 10000 == 0 and not os.path.exists(d):
				os.makedirs(d)
			z = p['url_z']
			print "%s - %s - %s" % (inc, p['id'], z)
			for e in xrange(10):
				time.sleep(0.5)
				f = '%s/%06d_%s.jpg' % (d, inc, p['id'])
				try:
					urllib.urlretrieve(z, f)
					inc += 1
					if inc == limit:
						sys.exit(0)
					break
				except e:
					if os.path.exists(f):
						os.remove(f)
					print e
	page += 1
