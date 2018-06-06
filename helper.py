def print_results(y_test, y_pred, name, acc, con_mat, class_rep):
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    print 'Results for {}:'.format(name)
    if acc:
        print 'Accuracy = {:.4f}'.format(accuracy_score(y_test,y_pred))
    if con_mat:
        print 'Confusion matrix:'
        print confusion_matrix(y_test, y_pred)
    if class_rep:
        print classification_report(y_test, y_pred)
    

def train_predict(X_train, X_test, y_train, y_test, models, acc=True, con_mat=True, class_rep=False):
    preds = {}
    for clf in models:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        preds[name] = clf.predict(X_test)
        
        print_results(y_test, preds[name], name, acc=acc, con_mat=con_mat, class_rep=class_rep)
 

def optimize_models(X_train, X_test, y_train, y_test, models, params, show_best_params = True):
    from sklearn.model_selection import GridSearchCV

    for model,param in zip(models,params):
        grid_obj = GridSearchCV(model, param)
        grid_fit = grid_obj.fit(X_train, y_train)
        best_clf = grid_fit.best_estimator_
        best_pred = best_clf.predict(X_test)
        model_name = best_clf.__class__.__name__
        
        best_pred_prob = best_clf.predict_proba(X_test)[:,1]
        
        print_results(y_test, best_pred, model_name, acc=True,con_mat=True,class_rep=False)
        if show_best_params:
            print grid_fit.best_params_
        