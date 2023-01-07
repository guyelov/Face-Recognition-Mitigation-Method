from math import sqrt


def calcRandomMse(origRatingsDict, randomMethod, numOfRandomRatings):
    mse = 0

    if randomMethod == 'uniform':
    # calculating uniformly random ratings:
        for key in origRatingsDict:
            for i in range(numOfRandomRatings):
                randomRating = random.uniform(1, 5)
                print(origRatingsDict[key])
                mse += (randomRating - origRatingsDict[key]['original rating']) ** 2

    if randomMethod == 'stratified':
    # calculating a permutation of the original ratings:
    # hint - shuffle the indices of the original values.
        for key in origRatingsDict:
            originalRating = origRatingsDict[key]['original rating']
            keys = list(origRatingsDict.keys())
            random.shuffle(keys)
            for i in range(numOfRandomRatings):
                randomRating = origRatingsDict[keys[i]]['original rating']
                mse += (randomRating - originalRating) ** 2

    for rIdx in range(numOfRandomRatings):
    # calculating the square of the difference between the original and ranrom value and add it to mse variable:
        mse += (origRatingsDict[key]['original rating'] - randomRating) ** 2
    # averaging the sum of the square differences:
    mse = mse / (len(origRatingsDict) * numOfRandomRatings)

    return mse



def getMovieRecommendation(ratingsDF, userID, movieID, normalizeRatings=False):
    userIDlist = list(ratingsDF.index)
    numOfOtherRated = 0
    maxAbsRate = -np.infty

    # calculate user's ratings average:
    userAvg = 0
    numOfUserRated = 0
    for movieID in ratingsDF.columns:
        movieRatingByUser = list(ratingsDF[movieID].loc[[userID]])[0]
        if movieRatingByUser != '?':
            userAvg += movieRatingByUser
            numOfUserRated += 1
    userAvg = userAvg / numOfUserRated


    # Gets recommendations for a person by using a weighted average of every other user's rankings
    total = 0
    rankings_list = []
    for otherID in userIDlist:

        # don't compare me to myself
        if otherID == userID:
            continue

        sim = pearson_correlation_from_DF(userID, otherID, ratingsDF)
        print('similarity of user ' + str(userID) + ' with user ' + str(otherID) + ' is: ' + str(sim))

        # calculate other user's ratings average:
        otherAvg = 0
        numOfOtherRated = 0
        for movieID in ratingsDF.columns:
            movieRatingByOther = list(ratingsDF[movieID].loc[[otherID]])[0]
            if movieRatingByOther != '?':
                otherAvg += movieRatingByOther
                numOfOtherRated += 1
        otherAvg = otherAvg / numOfOtherRated


        movieRatingByOther = list(ratingsDF[movieID].loc[[otherID]])[0]
        movieRatingByUser = list(ratingsDF[movieID].loc[[userID]])[0]

        # if other user has rated current movie
        if movieRatingByOther != '?':
            numOfOtherRated += 1

            # normalize other's rating of movieID:
            if normalizeRatings:
                movieRatingByOther = movieRatingByOther - otherAvg


            # Similarity * score
            total += sim * movieRatingByOther


    # calculate recommendation:
    if numOfOtherRated == 0:
        return 0
    else:
        if normalizeRatings:
            return userAvg + (total / numOfOtherRated)
        else:
            return total / numOfOtherRated




def pearson_correlation_from_DF(userID, otherID, ratingsDF):
    # To get both rated items
    both_rated = {}
    for movieID in ratingsDF.columns:
        movieRatingByOther = list(ratingsDF[movieID].loc[[otherID]])[0]
        movieRatingByUser = list(ratingsDF[movieID].loc[[userID]])[0]

        if (movieRatingByOther != '?') and (movieRatingByUser != '?'):
            both_rated[movieID] = 1

    number_of_ratings = len(both_rated)

    # Checking for number of ratings in common
    if number_of_ratings == 0:
        return 0

    # Add up all the preferences of each user
    user_preferences_sum = sum([list(ratingsDF[movieID].loc[[userID]])[0] for movieID in both_rated])

    # Sum up the squares of preferences of each user
    user_square_preferences_sum = sum([pow(list(ratingsDF[movieID].loc[[userID]])[0], 2) for movieID in both_rated])

    # Sum up the product value of both preferences for each item
    product_sum_of_both_users = sum([list(ratingsDF[movieID].loc[[userID]])[0] * list(ratingsDF[movieID].loc[[otherID]])[0] for movieID in both_rated])

    # Calculate the pearson score
    numerator_value = product_sum_of_both_users - (user_preferences_sum * sum([list(ratingsDF[movieID].loc[[otherID]])[0] for movieID in both_rated]) / number_of_ratings)
    if numerator_value == 0:
        return 0
    denominator_value = sqrt((user_square_preferences_sum - pow(user_preferences_sum, 2) / number_of_ratings) * (sum([pow(list(ratingsDF[movieID].loc[[otherID]])[0], 2) for movieID in both_rated]) - pow(sum([list(ratingsDF[movieID].loc[[otherID]])[0] for movieID in both_rated]), 2) / number_of_ratings))
    if denominator_value == 0:
        return 0
    else:
        r = numerator_value / denominator_value
        return r