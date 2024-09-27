#! /bin/bash

instances_unresolved=(
# "django__django-10554"
# "django__django-10914"
# "django__django-10999"
# "django__django-11087"
# "django__django-11138"
# "django__django-11149"
# "django__django-11206"
# "django__django-11211"
# "django__django-11239"
# "django__django-11265"
# "django__django-11276"
# "django__django-11299"
# "django__django-11333"
# "django__django-11400"
# "django__django-11477"
# "django__django-11532"
# "django__django-11728"
# "django__django-11734"
# "django__django-11740"
"django__django-11815"
"django__django-11820"
"django__django-11848"
"django__django-11885"
"django__django-12125"
"django__django-12193"
"django__django-12209"
"django__django-12273"
"django__django-12325"
"django__django-12406"
"django__django-12663"
"django__django-12713"
"django__django-12754"
"django__django-12774"
"django__django-12858"
"django__django-12965"
"django__django-13028"
"django__django-13033"
"django__django-13112"
"django__django-13121"
"django__django-13128"
"django__django-13158"
"django__django-13195"
"django__django-13212"
"django__django-13297"
"django__django-13344"
"django__django-13346"
"django__django-13406"
"django__django-13449"
"django__django-13512"
"django__django-13513"
"django__django-13568"
"django__django-13786"
"django__django-13794"
"django__django-13809"
"django__django-13821"
"django__django-13925"
"django__django-13964"
"django__django-14007"
"django__django-14011"
"django__django-14017"
"django__django-14034"
"django__django-14122"
"django__django-14140"
"django__django-14155"
"django__django-14170"
"django__django-14311"
"django__django-14315"
"django__django-14351"
"django__django-14376"
"django__django-14404"
"django__django-14534"
"django__django-14580"
"django__django-14608"
"django__django-14631"
"django__django-14672"
"django__django-14725"
"django__django-14771"
"django__django-14792"
"django__django-15022"
"django__django-15037"
"django__django-15098"
"django__django-15127"
"django__django-15161"
"django__django-15252"
"django__django-15268"
"django__django-15280"
"django__django-15375"
"django__django-15382"
"django__django-15503"
"django__django-15525"
"django__django-15554"
"django__django-15563"
"django__django-15569"
"django__django-15629"
"django__django-15695"
"django__django-15814"
"django__django-15851"
"django__django-15916"
"django__django-15957"
"django__django-15973"
"django__django-15987"
"django__django-16032"
"django__django-16082"
"django__django-16100"
"django__django-16136"
"django__django-16256"
"django__django-16263"
"django__django-16315"
"django__django-16454"
"django__django-16502"
"django__django-16560"
"django__django-16631"
"django__django-16667"
"django__django-16877"
"django__django-16938"
"django__django-16950"
"django__django-17084"
"django__django-17087"
"pylint-dev__pylint-4551"
"pylint-dev__pylint-4604"
"pylint-dev__pylint-4661"
"pylint-dev__pylint-7080"
"pylint-dev__pylint-7277"
"pylint-dev__pylint-8898"
"pytest-dev__pytest-10051"
"pytest-dev__pytest-10081"
"pytest-dev__pytest-10356"
"pytest-dev__pytest-5262"
"pytest-dev__pytest-5631"
"pytest-dev__pytest-5787"
"pytest-dev__pytest-5840"
"pytest-dev__pytest-6197"
"pytest-dev__pytest-7236"
"pytest-dev__pytest-7324"
"pytest-dev__pytest-7490"
"pytest-dev__pytest-7521"
"sphinx-doc__sphinx-10323"
"sphinx-doc__sphinx-10435"
"sphinx-doc__sphinx-10449"
"sphinx-doc__sphinx-10614"
"sphinx-doc__sphinx-11445"
"sphinx-doc__sphinx-11510"
"sphinx-doc__sphinx-7440"
"sphinx-doc__sphinx-7454"
"sphinx-doc__sphinx-7462"
"sphinx-doc__sphinx-7590"
"sphinx-doc__sphinx-7748"
"sphinx-doc__sphinx-7757"
"sphinx-doc__sphinx-7889"
"sphinx-doc__sphinx-7910"
"sphinx-doc__sphinx-7985"
"sphinx-doc__sphinx-8035"
"sphinx-doc__sphinx-8056"
"sphinx-doc__sphinx-8120"
"sphinx-doc__sphinx-8265"
"sphinx-doc__sphinx-8269"
"sphinx-doc__sphinx-8459"
"sphinx-doc__sphinx-8548"
"sphinx-doc__sphinx-8551"
"sphinx-doc__sphinx-8593"
"sphinx-doc__sphinx-8621"
"sphinx-doc__sphinx-8638"
"sphinx-doc__sphinx-9229"
"sphinx-doc__sphinx-9230"
"sphinx-doc__sphinx-9258"
"sphinx-doc__sphinx-9281"
"sphinx-doc__sphinx-9320"
"sphinx-doc__sphinx-9461"
"sphinx-doc__sphinx-9591"
"sphinx-doc__sphinx-9602"
"sphinx-doc__sphinx-9658"
"sphinx-doc__sphinx-9673"
"sympy__sympy-12419"
"sympy__sympy-12481"
"sympy__sympy-12489"
"sympy__sympy-13031"
"sympy__sympy-13091"
"sympy__sympy-13551"
"sympy__sympy-13615"
"sympy__sympy-13757"
"sympy__sympy-13798"
"sympy__sympy-13852"
"sympy__sympy-13877"
"sympy__sympy-13878"
"sympy__sympy-13974"
"sympy__sympy-14248"
"sympy__sympy-14531"
"sympy__sympy-14976"
"sympy__sympy-15017"
"sympy__sympy-15345"
"sympy__sympy-15599"
"sympy__sympy-15976"
"sympy__sympy-16597"
"sympy__sympy-16792"
"sympy__sympy-17318"
"sympy__sympy-17630"
"sympy__sympy-18189"
"sympy__sympy-18199"
"sympy__sympy-18211"
"sympy__sympy-18698"
"sympy__sympy-19040"
"sympy__sympy-19495"
"sympy__sympy-19783"
"sympy__sympy-20428"
"sympy__sympy-20438"
"sympy__sympy-20590"
"sympy__sympy-20916"
"sympy__sympy-21379"
"sympy__sympy-21596"
"sympy__sympy-21612"
"sympy__sympy-21930"
"sympy__sympy-22080"
"sympy__sympy-23262"
"sympy__sympy-23413"
"sympy__sympy-24562"
)
instances_resolved=(
)

instances_left=()

# for dir in ../test_sphinx/*/; do
#     dir=${dir%*/}  # remove trailing slash
#     dir=${dir##*/}  # get only the directory name
#     instances_left+=("$dir")
# done

# Create a new array with elements from instances_left that are not in instances_resolved
instances=()
for instance in "${instances_left[@]}"; do
    if [[ ! " ${instances_resolved[*]} " =~ " ${instance} " ]]; then
        instances+=("$instance")
    fi
done

# instances=("${instances[@]:62}")
# Combine with instances_unresolved
# instances=("${instances_unresolved[@]}"  "${instances[@]:21}")
instances=("${instances_unresolved[@]}")

echo "Instances: ${instances[*]}"
echo "Number of instances: ${#instances[@]}"
# exit
instances_string=$(IFS=,; echo "${instances[*]}")

run_instance() {
    local instance=$1
    local run_id=$2
    LANGCHAIN_PROJECT=unresolved_$instance python benchmark_copy.py --test-instance-ids $instance --run-id $run_id
}

# Set the number of instances to run in parallel
k=1
run_id="langgraph_agent_$(date +%s%N)"
echo "Run ID: $run_id"
# Run instances in parallel, k at a time
for ((i=0; i<${#instances[@]}; i+=k)); do
    # Get up to k instances
    docker rmi $(docker images -f "dangling=true" -q)
    batch=("${instances[@]:i:k}")
    
    # Run the batch in parallel
    for instance in "${batch[@]}"; do
        run_instance "$instance" "$run_id" &
    done
    
    # Wait for all background processes to finish before starting the next batch
    wait
    docker rmi $(docker images -f "dangling=true" -q)
done