import matplotlib.pyplot as plt
import seaborn as sns

class visualization:

    sns.set_style("whitegrid")
    sns.set_palette("cubehelix")

    @classmethod
    def configure_plot(self, style, palette):
        self.style = style
        self.palette = palette
        sns.set_style(style)
        sns.set_palette(palette)

    @classmethod
    def histogram(self, data, bins):
        sns.histplot(x=x,y=y,bins=bins)
    
    @classmethod
    def barplot(self, x,y):
        sns.barplot(x,y)

    @classmethod
    def lineplot(self, x, y):
        sns.lineplot(x,y)

    @classmethod
    def scatterplot(self, x,y):
        sns.scatterplot(x, y)

    @classmethod
    def heatmap(self, data):
        sns.heatmap(data)