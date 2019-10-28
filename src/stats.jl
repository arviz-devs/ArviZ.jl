Base.summary(data::InferenceData) = Pandas.DataFrame(arviz.summary(data.o))
