// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package org.apache.doris.planner;

import org.apache.doris.analysis.Analyzer;
import org.apache.doris.analysis.ExplainOptions;
import org.apache.doris.analysis.Expr;
import org.apache.doris.analysis.InsertStmt;
import org.apache.doris.analysis.QueryStmt;
import org.apache.doris.analysis.SelectStmt;
import org.apache.doris.analysis.SlotDescriptor;
import org.apache.doris.analysis.SlotId;
import org.apache.doris.analysis.StatementBase;
import org.apache.doris.analysis.StorageBackend;
import org.apache.doris.analysis.TupleDescriptor;
import org.apache.doris.catalog.PrimitiveType;
import org.apache.doris.catalog.ScalarType;
import org.apache.doris.common.util.VectorizedUtil;
import org.apache.doris.common.UserException;
import org.apache.doris.common.profile.PlanTreeBuilder;
import org.apache.doris.common.profile.PlanTreePrinter;
import org.apache.doris.common.util.VectorizedUtil;
import org.apache.doris.qe.ConnectContext;
import org.apache.doris.rewrite.mvrewrite.MVSelectFailedException;
import org.apache.doris.thrift.TExplainLevel;
import org.apache.doris.thrift.TQueryOptions;
import org.apache.doris.thrift.TRuntimeFilterMode;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The planner is responsible for turning parse trees into plan fragments that can be shipped off to backends for
 * execution.
 */
public class Planner {
    private static final Logger LOG = LogManager.getLogger(Planner.class);

    private boolean isBlockQuery = false;

    protected ArrayList<PlanFragment> fragments = Lists.newArrayList();

    private PlannerContext plannerContext;
    private SingleNodePlanner singleNodePlanner;
    private DistributedPlanner distributedPlanner;

    public boolean isBlockQuery() {
        return isBlockQuery;
    }

    public List<PlanFragment> getFragments() {
        return fragments;
    }

    public PlannerContext getPlannerContext() { return plannerContext;}

    public List<ScanNode> getScanNodes() {
        if (singleNodePlanner == null) {
            return Lists.newArrayList();
        }
        return singleNodePlanner.getScanNodes();
    }

    public void plan(StatementBase queryStmt, Analyzer analyzer, TQueryOptions queryOptions)
            throws UserException {
        createPlanFragments(queryStmt, analyzer, queryOptions);
    }

    /**
     */
    private void setResultExprScale(Analyzer analyzer, ArrayList<Expr> outputExprs) {
        for (TupleDescriptor tupleDesc : analyzer.getDescTbl().getTupleDescs()) {
            for (SlotDescriptor slotDesc : tupleDesc.getSlots()) {
                for (Expr expr : outputExprs) {
                    List<SlotId> slotList = Lists.newArrayList();
                    expr.getIds(null, slotList);
                    if (PrimitiveType.DECIMALV2 != expr.getType().getPrimitiveType()) {
                        continue;
                            }

                    if (PrimitiveType.DECIMALV2 != slotDesc.getType().getPrimitiveType()) {
                        continue;
                            }

                    if (slotList.contains(slotDesc.getId()) && null != slotDesc.getColumn()) {
                        int outputScale = slotDesc.getColumn().getScale();
                        if (outputScale >= 0) {
                            if (outputScale > expr.getOutputScale()) {
                                expr.setOutputScale(outputScale);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Return combined explain string for all plan fragments.
     */
    public String getExplainString(List<PlanFragment> fragments, ExplainOptions explainOptions) {
        Preconditions.checkNotNull(explainOptions);
        if (explainOptions.isGraph()) {
            // print the plan graph
            PlanTreeBuilder builder = new PlanTreeBuilder(fragments);
            try {
                builder.build();
            } catch (UserException e) {
                LOG.warn("Failed to build explain plan tree", e);
                return e.getMessage();
            }
            return PlanTreePrinter.printPlanExplanation(builder.getTreeRoot());
        }

        // print text plan
        TExplainLevel explainLevel = explainOptions.isVerbose() ? TExplainLevel.VERBOSE : TExplainLevel.NORMAL;
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < fragments.size(); ++i) {
            PlanFragment fragment = fragments.get(i);
            if (i > 0) {
                // a blank line between plan fragments
                str.append("\n");
            }
            str.append("PLAN FRAGMENT " + i + "\n");
            str.append(fragment.getExplainString(explainLevel));
        }
        if (explainLevel == TExplainLevel.VERBOSE) {
            str.append(plannerContext.getRootAnalyzer().getDescTbl().getExplainString());
        }
        return str.toString();
    }

    /**
     * Create plan fragments for an analyzed statement, given a set of execution options. The fragments are returned in
     * a list such that element i of that list can only consume output of the following fragments j > i.
     */
    public void createPlanFragments(StatementBase statement, Analyzer analyzer, TQueryOptions queryOptions)
            throws UserException {
        QueryStmt queryStmt;
        if (statement instanceof InsertStmt) {
            queryStmt = ((InsertStmt) statement).getQueryStmt();
        } else {
            queryStmt = (QueryStmt) statement;
        }

        plannerContext = new PlannerContext(analyzer, queryStmt, queryOptions, statement);
        singleNodePlanner = new SingleNodePlanner(plannerContext);
        PlanNode singleNodePlan = singleNodePlanner.createSingleNodePlan();

        if (VectorizedUtil.isVectorized()) {
            singleNodePlan.convertToVectoriezd();
        }

        if (statement instanceof InsertStmt) {
            InsertStmt insertStmt = (InsertStmt) statement;
            insertStmt.prepareExpressions();
        }

        // TODO chenhao16 , no used materialization work
        // compute referenced slots before calling computeMemLayout()
        //analyzer.markRefdSlots(analyzer, singleNodePlan, resultExprs, null);

        setResultExprScale(analyzer, queryStmt.getResultExprs());

        // materialized view selector
        boolean selectFailed = singleNodePlanner.selectMaterializedView(queryStmt, analyzer);
        if (selectFailed) {
            throw new MVSelectFailedException("Failed to select materialize view");
        }

        /**
         * - Under normal circumstances, computeMemLayout() will be executed
         *     at the end of the init function of the plan node.
         * Such as :
         * OlapScanNode {
         *     init () {
         *         analyzer.materializeSlots(conjuncts);
         *         computeTupleStatAndMemLayout(analyzer);
         *         computeStat();
         *     }
         * }
         * - However Doris is currently unable to determine
         *     whether it is possible to cut or increase the columns in the tuple after PlanNode.init().
         * - Therefore, for the time being, computeMemLayout() can only be placed
         *     after the completion of the entire single node planner.
         */
        analyzer.getDescTbl().computeMemLayout();
        singleNodePlan.finalize(analyzer);
        
        if (queryOptions.num_nodes == 1) {
            // single-node execution; we're almost done
            singleNodePlan = addUnassignedConjuncts(analyzer, singleNodePlan);
            fragments.add(new PlanFragment(plannerContext.getNextFragmentId(), singleNodePlan,
                    DataPartition.UNPARTITIONED));
        } else {
            // all select query are unpartitioned.
            distributedPlanner = new DistributedPlanner(plannerContext);
            fragments = distributedPlanner.createPlanFragments(singleNodePlan);
        }

        // Optimize the transfer of query statistic when query doesn't contain limit.
        PlanFragment rootFragment = fragments.get(fragments.size() - 1);
        QueryStatisticsTransferOptimizer queryStatisticTransferOptimizer = new QueryStatisticsTransferOptimizer(rootFragment);
        queryStatisticTransferOptimizer.optimizeQueryStatisticsTransfer();

        // Create runtime filters.
        if (!ConnectContext.get().getSessionVariable().getRuntimeFilterMode().toUpperCase()
                .equals(TRuntimeFilterMode.OFF.name()) && !VectorizedUtil.isVectorized()) {
            RuntimeFilterGenerator.generateRuntimeFilters(analyzer, rootFragment.getPlanRoot());
        }

	    if (statement instanceof InsertStmt && !analyzer.getContext().isTxnModel()) {
            InsertStmt insertStmt = (InsertStmt) statement;
            rootFragment = distributedPlanner.createInsertFragment(rootFragment, insertStmt, fragments);
            rootFragment.setSink(insertStmt.getDataSink());
            insertStmt.complete();
            ArrayList<Expr> exprs = ((InsertStmt) statement).getResultExprs();
            List<Expr> resExprs = Expr.substituteList(
                    exprs, rootFragment.getPlanRoot().getOutputSmap(), analyzer, true);
            rootFragment.setOutputExprs(resExprs);
        } else {
            List<Expr> resExprs = Expr.substituteList(queryStmt.getResultExprs(),
                    rootFragment.getPlanRoot().getOutputSmap(), analyzer, false);
            rootFragment.setOutputExprs(resExprs);
        }
        LOG.debug("finalize plan fragments");
        for (PlanFragment fragment : fragments) {
            fragment.finalize(queryStmt);
        }

        Collections.reverse(fragments);

        pushDownResultFileSink(analyzer);

        if (queryStmt instanceof SelectStmt) {
            SelectStmt selectStmt = (SelectStmt) queryStmt;
            if (queryStmt.getSortInfo() != null || selectStmt.getAggInfo() != null) {
                isBlockQuery = true;
                LOG.debug("this is block query");
            } else {
                isBlockQuery = false;
                LOG.debug("this isn't block query");
            }
        }
    }

    /**
     * If there are unassigned conjuncts, returns a SelectNode on top of root that evaluate those conjuncts; otherwise
     * returns root unchanged.
     */
    private PlanNode addUnassignedConjuncts(Analyzer analyzer, PlanNode root)
            throws UserException {
        Preconditions.checkNotNull(root);
        // List<Expr> conjuncts = analyzer.getUnassignedConjuncts(root.getTupleIds());

        List<Expr> conjuncts = analyzer.getUnassignedConjuncts(root);
        if (conjuncts.isEmpty()) {
            return root;
        }
        // evaluate conjuncts in SelectNode
        SelectNode selectNode = new SelectNode(plannerContext.getNextNodeId(), root, conjuncts);
        selectNode.init(analyzer);
        Preconditions.checkState(selectNode.hasValidStats());
        return selectNode;
    }

    /**
     * This function is mainly used to try to push the top-level result file sink down one layer.
     * The result file sink after the pushdown can realize the function of concurrently exporting the result set.
     * Push down needs to meet the following conditions:
     * 1. The query enables the session variable of the concurrent export result set
     * 2. The top-level fragment is not a merge change node
     * 3. The export method uses the s3 method
     *
     * After satisfying the above three conditions,
     * the result file sink and the associated output expr will be pushed down to the next layer.
     * The second plan fragment performs expression calculation and derives the result set.
     * The top plan fragment will only summarize the status of the exported result set and return it to fe.
     */
    private void pushDownResultFileSink(Analyzer analyzer) {
        if (fragments.size() < 1) {
            return;
        }
        if (!(fragments.get(0).getSink() instanceof ResultFileSink)) {
            return;
        }
        if (!ConnectContext.get().getSessionVariable().isEnableParallelOutfile()) {
            return;
        }
        if (!(fragments.get(0).getPlanRoot() instanceof ExchangeNode)) {
            return;
        }
        PlanFragment topPlanFragment = fragments.get(0);
        ExchangeNode topPlanNode = (ExchangeNode) topPlanFragment.getPlanRoot();
        // try to push down result file sink
        if (topPlanNode.isMergingExchange()) {
            return;
        }
        PlanFragment secondPlanFragment = fragments.get(1);
        ResultFileSink resultFileSink = (ResultFileSink) topPlanFragment.getSink();
        if (resultFileSink.getStorageType() == StorageBackend.StorageType.BROKER) {
            return;
        }
        if (secondPlanFragment.getOutputExprs() != null) {
            return;
        }
        // create result file sink desc
        TupleDescriptor fileStatusDesc = constructFileStatusTupleDesc(analyzer);
        resultFileSink.resetByDataStreamSink((DataStreamSink) secondPlanFragment.getSink());
        resultFileSink.setOutputTupleId(fileStatusDesc.getId());
        secondPlanFragment.setOutputExprs(topPlanFragment.getOutputExprs());
        secondPlanFragment.resetSink(resultFileSink);
        ResultSink resultSink = new ResultSink(topPlanNode.getId());
        topPlanFragment.resetSink(resultSink);
        topPlanFragment.resetOutputExprs(fileStatusDesc);
        topPlanFragment.getPlanRoot().resetTupleIds(Lists.newArrayList(fileStatusDesc.getId()));
    }

    /**
     * Construct a tuple for file status, the tuple schema as following:
     * | FileNumber | Int     |
     * | TotalRows  | Bigint  |
     * | FileSize   | Bigint  |
     * | URL        | Varchar |
     */
    private TupleDescriptor constructFileStatusTupleDesc(Analyzer analyzer) {
        TupleDescriptor resultFileStatusTupleDesc =
                analyzer.getDescTbl().createTupleDescriptor("result_file_status");
        resultFileStatusTupleDesc.setIsMaterialized(true);
        SlotDescriptor fileNumber = analyzer.getDescTbl().addSlotDescriptor(resultFileStatusTupleDesc);
        fileNumber.setLabel("FileNumber");
        fileNumber.setType(ScalarType.createType(PrimitiveType.INT));
        fileNumber.setIsMaterialized(true);
        fileNumber.setIsNullable(false);
        SlotDescriptor totalRows = analyzer.getDescTbl().addSlotDescriptor(resultFileStatusTupleDesc);
        totalRows.setLabel("TotalRows");
        totalRows.setType(ScalarType.createType(PrimitiveType.BIGINT));
        totalRows.setIsMaterialized(true);
        totalRows.setIsNullable(false);
        SlotDescriptor fileSize = analyzer.getDescTbl().addSlotDescriptor(resultFileStatusTupleDesc);
        fileSize.setLabel("FileSize");
        fileSize.setType(ScalarType.createType(PrimitiveType.BIGINT));
        fileSize.setIsMaterialized(true);
        fileSize.setIsNullable(false);
        SlotDescriptor url = analyzer.getDescTbl().addSlotDescriptor(resultFileStatusTupleDesc);
        url.setLabel("URL");
        url.setType(ScalarType.createType(PrimitiveType.VARCHAR));
        url.setIsMaterialized(true);
        url.setIsNullable(false);
        resultFileStatusTupleDesc.computeStatAndMemLayout();
        return resultFileStatusTupleDesc;
    }

    private static class QueryStatisticsTransferOptimizer {
        private final PlanFragment root;

        public QueryStatisticsTransferOptimizer(PlanFragment root) {
            Preconditions.checkNotNull(root);
            this.root = root;
        }

        public void optimizeQueryStatisticsTransfer() {
            optimizeQueryStatisticsTransfer(root, null);
        }

        private void optimizeQueryStatisticsTransfer(PlanFragment fragment, PlanFragment parent) {
            if (parent != null && hasLimit(parent.getPlanRoot(), fragment.getPlanRoot())) {
                fragment.setTransferQueryStatisticsWithEveryBatch(true);
            }
            for (PlanFragment child : fragment.getChildren()) {
                optimizeQueryStatisticsTransfer(child, fragment);
            }
        }

        // Check whether leaf node contains limit.
        private boolean hasLimit(PlanNode ancestor, PlanNode successor) {
            final List<PlanNode> exchangeNodes = Lists.newArrayList();
            collectExchangeNode(ancestor, exchangeNodes);
            for (PlanNode leaf : exchangeNodes) {
                if (leaf.getChild(0) == successor
                        && leaf.hasLimit()) {
                    return true;
                }
            }
            return false;
        }

        private void collectExchangeNode(PlanNode planNode, List<PlanNode> exchangeNodes) {
            if (planNode instanceof ExchangeNode) {
                exchangeNodes.add(planNode);
            }

            for (PlanNode child : planNode.getChildren()) {
                if (child instanceof ExchangeNode) {
                    exchangeNodes.add(child);
                } else {
                    collectExchangeNode(child, exchangeNodes);
                }
            }
        }
    }
}
