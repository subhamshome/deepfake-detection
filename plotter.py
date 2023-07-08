import seaborn as sns
import matplotlib.pyplot as plt


def corr_heatmap(X_train):
    X_train.corr()
    sns.heatmap(X_train.corr())


def scatter_plot(X_train, y_train):
    variances = X_train.var()
    sorted_variances = variances.sort_values(ascending=False)
    best_feature = sorted_variances.index[0]
    second_best_feature = sorted_variances.index[1]
    X_high_var = X_train.iloc[:, [best_feature, second_best_feature]]

    plt.scatter(X_high_var.iloc[:, 0],
                X_high_var.iloc[:, 1], c=y_train, cmap='viridis')
    plt.xlabel(f"Feature {best_feature}")
    plt.ylabel(f"Feature {second_best_feature}")
    plt.title('Scatter Plot of Features with Highest Variance')
    plt.show()


def roc_plot(fpr, tpr, acc):
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='AUC = %0.2f' % acc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Validation Dataset')
    plt.legend(loc="lower right")
    plt.show()
