movies = open('movie.txt', 'r')
users = open('user.txt', 'r')

movieDict = {
    'id': {

    },
    'year': {

    },
    'genre': {

    }
}

moviesLength = 0
for movie in movies:
    moviesLength += 1
    splitMovie = movie.strip('\n').split(',')
    if splitMovie[0] in movieDict['id']:
        movieDict['id'][splitMovie[0]] += 1
    else:
        movieDict['id'][splitMovie[0]] = 1
    if splitMovie[1] in movieDict['year']:
        movieDict['year'][splitMovie[1]] += 1
    else:
        movieDict['year'][splitMovie[1]] = 1
    if splitMovie[2] in movieDict['genre']:
        movieDict['genre'][splitMovie[2]] += 1
    else:
        movieDict['genre'][splitMovie[2]] = 1
print 'LIST LENGTH >>>>>: %s' % moviesLength
print 'EMPTY YEARS >>>>>: %s, %s%%' % (movieDict['year']['N/A'], int(movieDict['year']['N/A']/float(moviesLength - 1) * 100))
print 'EMPTY GENRES >>>>>: %s, %s%%' % (movieDict['genre']['N/A'], int(movieDict['genre']['N/A']/float(moviesLength - 1) * 100))