def train_predict(X_train, X_test, y_train, y_test, models, acc=True, con_mat=True, class_rep=False):
    from sklearn.metrics import accuracy_score, confusion_matrix
    preds = {}
    for clf in models:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        preds[name] = clf.predict(X_test)  
        print 'Results for {}:'.format(name)
        if acc:
            print 'Accuracy = {:.4f}'.format(accuracy_score(y_test, preds[name]))
        if con_mat:
            print 'Confusion matrix:'
            print confusion_matrix(y_test, preds[name])
        if class_rep:
            print classification_report(y_test, preds[name])