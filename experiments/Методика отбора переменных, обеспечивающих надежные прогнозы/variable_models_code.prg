
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''Как запустить код'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''Этот код работает с квартальными или месячными данными.
'''Убедитесь, что начальная дата wf совпадает с датой первого наблюдения "y".
'''убедитесь, что между первым и последним наблюдением данных y и x нет пропущенных значений.
''' вставьте данные в стационарном виде в программу E-views wf.
''' назовите зависимую переменную "y". 
''' для переменных "x", Quick> Empty Group. вставьте переменные "x" без имен и закройте группу без сохранения. Eviews назовет серию переменных "x" как "ser*".
''' Запустите код в quietly режиме для получения более быстрых результатов. 


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''Результат выполнения кода''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''' Показатели по критерию "results1" (RMSE-sum) представлены в "Таблице результатов1" (подробнее см. раздел 8).
'''результаты по критерию "results2" (outperform ratio и т.д.) представлены в "Table Results2" (подробнее см. раздел 8).
''' результаты по переменным "nbrbest" (например, 5) представлены в "Таблице Результаты Best"
'''RRMSE лучших "nbrbest" переменных по "results1" и "results2", а также рекурсивные прогнозы этих переменных представлены графически (подробнее см. раздел 11).



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''1) Система оценки'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''общий вид моделиl: y=c+b1*L(y)+b2*L(x)+e.
'''лаги переменных y и x определяются в соответствии с критерием Шварца

close @objects
delete  bench* mod* num* gro* na* obs*   f_* hor* s_* sf* v_* rmse* m_* var* result* sele* sil* minlag* outper* rks* lastobs* stepsi* nbr* ini* best* graph* recur* g_* x*


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''2) Структура прогнозирования'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''определить максимальное количество лагов для y;
!maxlagy					=4
'''определить максимальное количество лагов для x;
!maxlagx					=4
'''определить минимальное количество лагов для x;
!minlagx					=1
'''определить максимальное количество наблюдений в первой рекурсивной оценке;
scalar numobs			=31
''определить горизонт прогнозирования;
scalar horizon			=4
'''определение размера шага рекурсивного оценивания;
scalar stepsize			=1
'''определить процент, на который переменная превосходит эталонный показатель хотя бы на;
scalar outperformby	=10
'''определить количество наилучших переменных для построения графика;
scalar nbrbest			=3

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''3) Исходные условия''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
smpl @all
'''A) групповые ряды для определения количества переменных.
group groupx ser*
stomna(groupx, x)
scalar numser=@columns(x)

'''B) определение последней даты каждого ряда X для рекурсивных оценок.
'''B.1) определить диапазон выборки, начиная с первого наблюдения до некоторой произвольной даты, чтобы определить недостающие значения в начале выборки.
smpl @all
matrix(numser,1) m_na_ser
for !h=1 to numser
	%b = @str(!h, "i02")
     scalar lastobs_{%b}=@ilast(ser{%b})
	smpl @first @first+lastobs_{%b}-1
     scalar na_ser{%b}=@nas(ser{%b})
	m_na_ser(!h,1)=na_ser{%b}
next

vector v_maxnbr_na=@cmax(m_na_ser)

if !maxlagy>!maxlagx then
	scalar initial=!maxlagy+v_maxnbr_na(1)
else
	scalar initial=!maxlagx+v_maxnbr_na(1)
endif

'''''B.2) для каждого ряда X, в соответствии с количеством наблюдений, определить последнюю дату, когда можно проводить оценку.
smpl @all
for !h=1 to numser
	%b = @str(!h, "i02")	
	scalar obsser{%b}=na_ser{%b}+@obs(ser{%b})
		if @obs(y)>obsser{%b}+!minlagx then
			scalar obsser{%b}=obsser{%b}+!minlagx
		else
			scalar obsser{%b}= @obs(y)
		endif
next


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''4) Эталонная оценка'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
smpl @all
!bestlagy		=0
!schwarz 		= 99999999
for !j=numobs to @obs(y)
	smpl @first+initial @first+!j-1
	for !i=1 to !maxlagy
		equation benchmark_!j_!i.ls(cov=hac) y c y(-!i to -1)
	next
	for !i=1 to !maxlagy		
		if benchmark_!j_!i.@schwarz<!schwarz then
				!bestlagy=!i
				!schwarz=benchmark_!j_!i.@schwarz
		endif
	next
	copy benchmark_!j_!bestlagy benchmark_!j
	!bestlagy=0
	!schwarz= 99999999
next


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''5) Оценка модели'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
smpl @all
	!bestlagx=0
	!bestlagy=0
	!schwarz= 99999999
for !h=1 to numser
	%b = @str(!h, "i02")		
	for !j=numobs to obsser{%b}
		smpl @first+initial @first+!j-1
		for !i=1 to !maxlagy
			for !k=!minlagx to !maxlagx
				equation mod_{%b}_!j_!i_!k.ls(cov=hac) y c y(-!i to -1) ser{%b}(-!k to -!minlagx)
				if mod_{%b}_!j_!i_!k.@schwarz<!schwarz then
					!bestlagx=!k
					!schwarz=mod_{%b}_!j_!i_!k.@schwarz
				endif			
			next
			!schwarz=999999999				
			mod_{%b}_!j_!i_!bestlagx.ls(cov=hac) y c y(-!i to -1) ser{%b}(-!bestlagx to -!minlagx)
			copy 	mod_{%b}_!j_!i_!bestlagx mod_{%b}_!j_!i
		next
		!schwarz=99999999
		for !i=1 to !maxlagy
			if mod_{%b}_!j_!i.@schwarz<!schwarz then
				!bestlagy=!i
				!schwarz=mod_{%b}_!j_!i.@schwarz
			endif
		next
		copy  mod_{%b}_!j_!bestlagy mod_{%b}_!j 
		!schwarz=99999999
		!bestlagx=0
		!bestlagy=0
	next
	!schwarz=99999999
	!bestlagx=0
	!bestlagy=0
next	


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''6) Прогноз'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''A) benchmark forecasts.
smpl @all
for !j=numobs to @obs(y)-horizon step stepsize
	smpl @first+!j @first+!j-1+horizon
	benchmark_!j.forecast f_benchmark_!j
	vector v_f_benchmark_!j=@convert(f_benchmark_!j)
	mtos( v_f_benchmark_!j,s_f_benchmark_!j)
next

'''B) model forecasts.
smpl @all
for !h=1 to numser 
	%b = @str(!h, "i02")	
	for !j=numobs to obsser{%b}-horizon step stepsize
		smpl @first+!j @first+!j-1+horizon 
		mod_{%b}_!j.forecast f_mod_{%b}_!j
		vector v_f_mod_{%b}_!j=@convert(f_mod_{%b}_!j)
		mtos(v_f_mod_{%b}_!j,s_f_mod_{%b}_!j)
	next
next


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''7) Расчет RMSE'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''A) rmse эталона.
smpl @all
for !j=numobs to @obs(y)-horizon step stepsize
	series rmse_benchmark_!j = @sqrt(@mean((s_f_benchmark_!j-y)^2))
	matrix(@obs(y)-horizon-numobs+1,1)  m_rmse_benchmark
	!row=!j-numobs+1
	m_rmse_benchmark(!row)= rmse_benchmark_!j(1)
next

''''''B) rmse моделей.
matrix(numser,1) m_obsser
for !h=1 to numser
	%b = @str(!h, "i02")	
	m_obsser(!h)=obsser{%b}
	for !j=numobs to obsser{%b}-horizon step stepsize
		series rmse_{%b}_!j=@sqrt(@mean((s_f_mod_{%b}_!j-y)^2))
		matrix(obsser{%b}-horizon-numobs+1,1)  m_rmse_{%b}
		matrix(obsser{%b}-horizon-numobs+1,1)  m_zero_{%b}
		!row=!j-numobs+1
		m_rmse_{%b}(!row)= rmse_{%b}_!j(1)
	next
next

''''''C) вычисление суммы RMSE.
vector v_obsser_min=@cmin(m_obsser)
matrix(v_obsser_min(1)-horizon-numobs+1,1) m_rmse_obsminbench
for !h=1 to numser
	%b = @str(!h, "i02")
	matrix(v_obsser_min(1)-horizon-numobs+1,1) m_rmse_obsmin_{%b}
	for !row=1 to v_obsser_min(1)-horizon-numobs+1
		m_rmse_obsmin_{%b}(!row)=m_rmse_{%b}(!row)
		m_rmse_obsminbench(!row)=m_rmse_benchmark(!row)
	next
	vector v_rmsesum_obsmin_{%b}=@csum(m_rmse_obsmin_{%b})
	vector v_rmsesum_obsminbench=@csum(m_rmse_obsminbench)
next


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''8) Сортировка переменных'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''A) results1; сортировка переменных по их сумме среднеквадратичных отклонений рекурсивных оценок относительно суммы среднеквадратичных отклонений эталонных оценок.
matrix(numser,2) m_rrmsesum_inter
matrix(numser,2) m_rrmsesum_sorted

for !a=1 to 2
	table(numser+1,6) results{!a}
next
results1(1,1)="номер переменной"
results1(1,2)="сумма среднеквадратичных отклонений относительно эталона"
for !h=1 to numser
	%b = @str(!h, "i02")	
	m_rrmsesum_inter(!h,1)={%b}
	m_rrmsesum_inter(!h,2)=v_rmsesum_obsmin_{%b}(1)/v_rmsesum_obsminbench(1)
next

vector rks_rrmsesum=@ranks(@columnextract(m_rrmsesum_inter,2),"a","i")
m_rrmsesum_sorted=@capplyranks(m_rrmsesum_inter,rks_rrmsesum)
for !h=1 to numser
	for !column=1 to 2
	results1(!h+1,!column)=m_rrmsesum_sorted(!h,!column)
	next
next

'''B) results2, step1 ;  определение того, сколько раз каждая переменная выигрывает у эталона, выигрывает не менее чем на ... процентов, сколько раз она дает лучший и худший прогнозы.
'''B.1)
matrix(numser,6) m_marcellino_t4
for !h=1 to numser
	%b = @str(!h, "i02")	
	m_marcellino_t4(!h,1)={%b}
	for !nbrinter=1 to 2
		matrix(obsser{%b}-horizon-numobs+1,1) m_bench_inter{!nbrinter}_{%b}
	next
	for !row=1 to obsser{%b}-horizon-numobs+1
		m_bench_inter1_{%b}(!row)=m_rmse_benchmark(!row)
		m_bench_inter2_{%b}(!row)=m_rmse_benchmark(!row)*(1-(outperformby)/100)
	next
	for !out=1 to 2
		matrix m_outperform{!out}_{%b}=@elt(m_rmse_{%b},m_bench_inter{!out}_{%b})
		vector v_outperform{!out}_{%b}=@csum(m_outperform{!out}_{%b})
	next
	m_marcellino_t4(!h,2)=@csum(@egt(m_rmse_{%b},m_zero_{%b}))(1)
	m_marcellino_t4(!h,3)=v_outperform1_{%b}(1)
	m_marcellino_t4(!h,4)=v_outperform2_{%b}(1)
next

''''B.2)определение того, сколько раз каждая переменная дает наилучший и наихудший прогноз.
matrix(@obs(y)-horizon-numobs+1,numser) m_na_inter=na
for !h=1 to numser
	%b = @str(!h, "i02")		
	matrix(obsser{%b}-horizon-numobs+1,1)  m_rrmse_{%b}
	for !row= 1 to obsser{%b}-horizon-numobs+1
		if m_bench_inter1_{%b}(!row)<>0 then 
			m_rrmse_{%b}(!row)=m_rmse_{%b}(!row)/m_bench_inter1_{%b}(!row)
		else
			m_rrmse_{%b}(!row)=0
		endif
	next
	matplace(m_na_inter,m_rrmse_{%b},1,!h)
next
matrix m_na_inter_trans=@transpose(m_na_inter)
for !row=1 to @obs(y)-horizon-numobs+1
	vector(@obs(y)-horizon-numobs+1) v_varnbr_minrrmse
	vector(@obs(y)-horizon-numobs+1) v_varnbr_maxrrmse
	vector v_intermin_!row=@cmin(@columnextract(m_na_inter_trans,!row))
	vector v_intermax_!row=@cmax(@columnextract(m_na_inter_trans,!row))
	vector v_intercimin_!row=@cimin(@columnextract(m_na_inter_trans,!row))
	vector v_intercimax_!row=@cimax(@columnextract(m_na_inter_trans,!row))
	if v_intermin_!row(1)<>0 then
		v_varnbr_minrrmse(!row)=v_intercimin_!row(1)
		v_varnbr_maxrrmse(!row)=v_intercimax_!row(1)	
	else
		v_varnbr_minrrmse(!row)=0
		v_varnbr_maxrrmse(!row)=0
	endif
next

'''B.3)соответствие количества лучших и худших прогнозов переменным.
mtos(v_varnbr_minrrmse, s_varnbr_minrrmse)
mtos(v_varnbr_maxrrmse, s_varnbr_maxrrmse)
series s_inter_min=@obsby(s_varnbr_minrrmse,s_varnbr_minrrmse)
series s_inter_max=@obsby(s_varnbr_maxrrmse,s_varnbr_maxrrmse)
matrix(@obs(y)-horizon-numobs+1,2) m_inter_min
matrix(@obs(y)-horizon-numobs+1,2) m_inter_max
for !row=1 to @obs(y)-horizon-numobs+1
	m_inter_min(!row,1)=v_varnbr_minrrmse(!row)
	m_inter_min(!row,2)=s_inter_min(!row)
	m_inter_max(!row,1)=v_varnbr_maxrrmse(!row)
	m_inter_max(!row,2)=s_inter_max(!row)
next

for !h=1 to numser
	for !row=1 to @obs(y)-horizon-numobs+1
		if m_marcellino_t4(!h,1)=m_inter_min(!row,1) then 
			m_marcellino_t4(!h,5)=m_inter_min(!row,2) 
		endif
		if m_marcellino_t4(!h,1)=m_inter_max(!row,1) then 
			m_marcellino_t4(!h,6)=m_inter_max(!row,2) 
		endif		
	next
next

'''B.4) перенос результатов матрицы в results2;
results2(1,1)="variable number"
results2(1,2)="number of recursive estimations"
results2(1,3)="nbr of times the variable outperforms benchmark"
results2(1,4)= "nbr of times the variable outperforms benchmark by at least " + @str(outperformby) + "%"
results2(1,5)= "nbr of times the variable produces best forecasts"
results2(1,6)= "nbr of times the variable produces worst forecasts"
for !h= 1 to numser
	for !column=1 to 6
		results2(!h+1,!column)=m_marcellino_t4(!h,!column)
	next
next


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''9) Общие итоги'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
for !a=1 to 2
	show results!a
next


'''''''''''''''''''''''''''''''''''''''''''''''''''10) Результаты наилучших переменных''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''лучшие переменные "nbrbest" сводятся в таблицу (results_of_best). Во втором столбце таблицы переменные выбираются в соответствии с их коэффициентами превзошевшими эталон в results2.
delete m_su* rks_m_su* results_of*
matrix(numser,2) m_sub_marcellino
for !row=1 to numser
	m_sub_marcellino(!row,1)=m_marcellino_t4(!row,1)
	m_sub_marcellino(!row,2)=m_marcellino_t4(!row,3)/m_marcellino_t4(!row,2)
next

matrix(numser,2) m_sub_mar_sorted 
vector rks_m_sub_mar=@ranks(@columnextract(m_sub_marcellino,2),"d","i")
m_sub_mar_sorted=@capplyranks(m_sub_marcellino,rks_m_sub_mar)

table(nbrbest+1,3) results_of_best
results_of_best(1,1)="variable number of top " + @str(nbrbest) + " acc. to results1"
results_of_best(1,2)="variable number of top " + @str(nbrbest) + " acc. to results2"
results_of_best(1,3)="outperform over estimation ratio acc. to results2"
for !row=2 to nbrbest+1
	results_of_best(!row,1)=results1(!row,1)
	results_of_best(!row,2)=m_sub_mar_sorted(!row-1,1)
	results_of_best(!row,3)=m_sub_mar_sorted(!row-1,2)
next
show results_of_best


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''11) Графики''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''A) удаление нулей, если они есть, из матриц m_rrmse*.
for !h=1 to numser
	%b = @str(!h, "i02")	
	scalar recur_nbr_{%b}=m_marcellino_t4(!h,2)
	matrix(recur_nbr_{%b},1) results_rrmse_{%b}
	!row1=1	
	!row2=1
	while !row1<=recur_nbr_{%b} and !row2<=obsser{%b}
		results_rrmse_{%b}(!row1)=m_rrmse_{%b}(!row2)
		!row1=!row1+1
		!row2=!row2+stepsize
	wend
next

''B) построение графика лучшей переменной "nbrbest" по "результатам1" и "результатам2".

for !a=1 to nbrbest
	scalar best_var_results1_!a=m_rrmsesum_sorted(!a,1)
	scalar best_var_results2_!a=m_sub_mar_sorted(!a,1)
	for !b=1 to 2
		if best_var_results{!b}_!a<10 then
			string best= "0"+ @str(best_var_results{!b}_!a)
			freeze(g_rrmse_bestres!b_{best}) results_rrmse_{best}.line
			group best_res!b_{best} f_mod_{best}* y
			freeze(g_forecast_bestres!b_{best}) best_res!b_{best}.line
				if nbrbest<=5 then
					show g_rrmse_bestres!b_{best}
					show g_forecast_bestres!b_{best}	
				endif
		else
			string best=@str(best_var_results{!b}_!a)
			freeze(g_rrmse_bestres!b_{best}) results_rrmse_{best}.line
			group best_res!b_{best} f_mod_{best}* y
			freeze(g_forecast_bestres!b_{best}) best_res!b_{best}.line
				if nbrbest<=5 then			
					show g_rrmse_bestres!b_{best}
					show g_forecast_bestres!b_{best}
				endif
		endif
	next
next


