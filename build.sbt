name := "jug-spark-mllib"

version := "1.0"

scalaVersion := "2.11.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.3.1"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "1.3.1"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.3.1"

libraryDependencies += "com.typesafe" % "config" % "1.2.1"

libraryDependencies += "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test"

libraryDependencies += "junit" % "junit" % "4.10" % "test"

libraryDependencies += "com.typesafe.scala-logging" % "scala-logging-slf4j_2.11" % "2.1.2"


fork := true