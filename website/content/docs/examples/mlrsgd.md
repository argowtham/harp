# Multiclass Logistic Regression

![Decision surface of Multiclass Logistic Regression](/img/4-3-1.png){width=60%}

Logistic Regression is a binary classifier which discriminates between target variable 1 or 0. It is a probabilistic based classifier where the function used is sigmoid function. Multiclass logistic regression (MLR) is a classification method which generalizes the logistic regression to multiclass problems, i.e., with more than two possible outcomes. MLR model predicts the probabilities of the different possible outcomes of a categorically distributed dependent variable, given a set of independent variables.

## Logistic Regression
The algorithm for logistic regression model is as follows:  
1. Initialize `w` weight vector randomly  
2. Predict the probability of success or `P(Y=1)` using the following formula  
![Sigmoid function](/img/4-3-3.png)  
where  
`w` - Weight vector  &  
`x` - Data point  
3. Compare the predicted output and the actual output.  
4. Based on the error, compute the slope and use Gradient descent to approximate `w`  
5. Repeat steps 2-4 until there is no significant improvement in reducing the error rate  

## One vs Rest Methodology
Here, we are adopting One vs Rest (OVR) methodology for performing multiclass logistic regression. This makes the weight vectors independent of each other and helps us in achieving parallelism in easier way. If our data has *'m'* classes and *'d'* dimensions, then our weight vector will be a *m\*d* matrix where *'m'* is the number of rows and *'d'* is the number of columns. The matrix is as shown below.  
![Weight matrix](/img/4-3-4.png)

## Stochastic Gradient descent
Stochastic gradient descent is the stochastic approximation of the gradient descent in which we take a random sample and use it for minimizing our objective function (error) which is written as a sum of partial derivatives. In other words, SGD tries to find minimum (descent) or maximum (ascent) by iteration. Normally, stochastic gradient can be slow to converge as it takes only one sample. Here, we try to improve on stochastic gradient by using thread level parallelism. This is achieved by the use of **Dynamic Scheduler**. As the algorithm sweeps through the training set, it performs the update for each training example. Several passes can be made over the training set until the algorithm converges.  

The steps of the SGD algorithm are described as follows:  
1. Randomly choose an initial weight vector `w` and learning rate parameter `r`   
2. Repeat until there is no improvement in prediction performance or for `k` iterations   
	a. Select a data point randomly  
    b. Update the weights based on the slope computed from partial derivatives  

#### Notations
* `n` - Number of data points
* `d` - Number of features/dimensions
* `m` - Number of class labels
* `w` - *m\*d* weight matrix
* `k` - Number of iterations

## Parallel Design
1. What is the model? What kind of data structure is used?  
&nbsp;&nbsp;&nbsp;&nbsp;Weight matrix which contains the weight for every feature is the model output. Because an ovr(one-versus-rest) approach is adopted, each weight vector are independent of each other. The structure of the weight matrix is shown above which is *m\*d* matrix.  


2. What are the characteristics of the data dependency in model update computation, can updates run concurrently?  
&nbsp;&nbsp;&nbsp;&nbsp;Model update computation here is the SGD update, in which for each data point it should update the model directly. Since OVR approach is used, the weight vectors are independent and hence each weight vector can be updated concurrently.  
&nbsp;&nbsp;&nbsp;&nbsp;The weight updates are dependent on the local data that is present with that mapper and each mapper updates its corresponding weight vector.  


3. Which kind of parallelism scheme is suitable, data parallelism or model parallelism?  
&nbsp;&nbsp;&nbsp;&nbsp;**Data parallelism** is used here where the training data is split between every mapper and each mapper gets a chunk of the training data.  
&nbsp;&nbsp;&nbsp;&nbsp;Because the model updates are running concurrently in our implementation, **model parallelism** is a natural solution here. Each node gets one partition of the model, which then updates it in parallel.  
&nbsp;&nbsp;&nbsp;&nbsp;And furthermore, **thread level parallelism** can also follows this model parallelism pattern, in which each thread take a subset of partition and perform gradient descent update in parallel independently.


4. Which collective communication operations is suitable to synchronize model?  
&nbsp;&nbsp;&nbsp;&nbsp;Dynamic Scheduler is used to achieve thread level parallelism where each mapper node submits the SGD task to the queue and waits for model parameter update. **Rotate** collective communication method is used for intermodel synchronization.  

## DATAFLOW

![dataflow](/img/4-3-2.png)

## Step 0 --- Data preprocessing

&nbsp;&nbsp;&nbsp;&nbsp;Harp MLR will use the data in the vector format. Each vector in a file represented by the format `<did> [<fid>:<weight>]`:

* `<did>` is an unique document id
* `<fid>` is a positive feature id
* `<weight>` is the number feature value within document weight

&nbsp;&nbsp;&nbsp;&nbsp;After preprocessing, push the data set into HDFS by the following commands.
```bash
hdfs dfs -mkdir /input  
hdfs dfs -put input_data/* /input
```

## Step 1 --- Initialize

&nbsp;&nbsp;&nbsp;&nbsp;For Harp MLR, we will use dynamic scheduling as mentioned above. Before we set up the dynamic scheduler, we need to initialize the weight matrix `W`, which will be partitioned into `T` parts representing to `T` labels which means that each label belongs to one partition and is treated as an independent task.
```Java
private void initTable() {
    wTable = new Table(0, new DoubleArrPlus());
    for (int i = 0; i < topics.size(); i++)
        wTable.addPartition(new Partition(i, DoubleArray.create(TERM + 1, false)));
}
```

&nbsp;&nbsp;&nbsp;&nbsp;After that we can initialize the dynamic scheduler. Each thread will be treated as a worker and be added into the scheduler. The only thing that needs to be done is that tasks has to be submitted during the computation.
```Java
private void initThread() {
    GDthread = new LinkedList<>();
    for (int i = 0; i < numThread; i++)
        GDthread.add(new GDtask(alpha, data, topics, qrels));
    GDsch = new DynamicScheduler<>(GDthread);
}
```

## Step 2 --- Mapper Communication
&nbsp;&nbsp;&nbsp;&nbsp;In this process, we use `regroup` collective communication to distribute the model data partitions to the workers first. The workers will get almost the same number of partitions. Once every mapper gets its own part of model data, we start the dynamic scheduler. Every mapper submits a task to dynamic scheduler, each task being a stochastic gradient descent task. The gradient descent is speed up using thread level parallelism. Rather than using a single data point, we are using *l* data points where *l* is the number of threads used.
&nbsp;&nbsp;&nbsp;&nbsp;After the workers finish once with their partition, we use `rotate` operation to swap the partitions among workers. When finishing the all process, each worker should use its own data training the whole partition `K` times, of which `K` is the number of iteration. `Allgather` operation is used to collect the partial data from each mapper and to produce the final model. This combined final model is then shared with all the workers which is used for prediction. The Master node outputs the weight matrix `W`.

```Java
protected void mapCollective(KeyValReader reader, Context context) throws IOException, InterruptedException {
    LoadAll(reader);
    initTable();
    initThread();

    regroup("MLR", "regroup_wTable", wTable, new Partitioner(getNumWorkers()));

    GDsch.start();        
    for (int iter = 0; iter < ITER * numMapTask; iter++) {
        for (Partition par : wTable.getPartitions())
            GDsch.submit(par);
        while (GDsch.hasOutput())
            GDsch.waitForOutput();
            
        rotate("MLR", "rotate_" + iter, wTable, null);

        context.progress();
    }
    GDsch.stop();
        
    allgather("MLR", "allgather_wTable", wTable);

    if (isMaster())
        Util.outputData(outputPath, topics, wTable, conf);
    wTable.release();
}
```

## USAGE

```bash
$ hadoop jar harp-tutorial-app-1.0-SNAPSHOT.jar edu.iu.mlr.MLRMapCollective [alpha] [number of iteration] [number of features] [number of workers] [number of threads] [topic file path] [qrel file path] [input path in HDFS] [output path in HDFS]
#e.g. hadoop jar harp-tutorial-app-1.0-SNAPSHOT.jar edu.iu.mlr.MLRMapCollective 1.0 100 47236 2 16 /rcv1v2/rcv1.topics.txt /rcv1v2/rcv1-v2.topics.qrels /input /output
```

The output should be the weight matrix `W`.
