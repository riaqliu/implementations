import waterfall_chart
import matplotlib.pyplot as plt

def post_plot(selected_features, shapleys):
    scores = list(zip(selected_features, shapleys))
    scores.sort(key=lambda x:x[1], reverse=True)
    a,b = tuple(list(l) for l in zip(*scores))
    waterfall_chart.plot(a, b, formatting='{:,.3f}')
    plt.ylim(bottom=0)
    plt.show()


if __name__ == "__main__":
    selected_features = ['Attribute1', 'Attribute2', 'Attribute10', 'Attribute12', 'Attribute22', 'Attribute36', 'Attribute44', 'Attribute46', 'Attribute48', 'Attribute57']
    shapleys = [0.026343253968254043, 0.01740323565323574, 0.14544619963369945, 0.08280692918192918, 0.06705792124542126, 0.10824000305250296, 0.1053687423687423, 0.08850366300366298, 0.12237919719169706, 0.04359371184371188]
    post_plot(selected_features, shapleys)