
# A2 - WEIGHTED kNN


def weighted_vote(neighbors):

    labels = []
    weights = []

    for neighbor in neighbors:

        distance_value = neighbor[0]

        label = neighbor[1]

        if distance_value == 0:

            weight = 1000000

        else:

            weight = 1 / distance_value

        found = False

        for i in range(len(labels)):

            if labels[i] == label:

                weights[i] += weight

                found = True

                break

        if not found:

            labels.append(label)

            weights.append(weight)

    highest_weight = weights[0]

    for weight in weights:

        if weight > highest_weight:

            highest_weight = weight

    tied_labels = []

    for i in range(len(labels)):

        if weights[i] == highest_weight:

            tied_labels.append(
                labels[i]
            )

    if len(tied_labels) == 1:

        return tied_labels[0]
    # closest neighbor among tied classes

    for neighbor in neighbors:

        for label in tied_labels:

            if neighbor[1] == label:

                return label
#exp

neighbors = [
    [0.5, 1],
    [1.0, 2],
    [2.0, 2]
]

prediction = weighted_vote(
    neighbors
)

print("A2 - WEIGHTED kNN")


print(
    "Weighted kNN Prediction =",
    prediction
)